"""Training orchestrator -- end-to-end pipeline with MLflow tracking."""

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from stroke_risk.data.loader import load_data
from stroke_risk.data.preprocessing import (
    build_preprocessor,
    get_feature_names,
    resample_data,
    split_data,
)
from stroke_risk.features.engineering import engineer_features
from stroke_risk.models.evaluate import (
    build_comparison_table,
    compute_metrics,
    find_optimal_threshold,
    get_classification_report_str,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from stroke_risk.models.optimize import optimize_all_models
from stroke_risk.models.stacking import train_stacking_ensemble
from stroke_risk.utils.config import load_all_configs

logger = logging.getLogger(__name__)


def run_training_pipeline(config_dir: str = "configs") -> dict[str, Any]:
    """Run the full training pipeline.

    Steps:
    1. Load and engineer features.
    2. Split and preprocess data.
    3. Resample training data.
    4. Optimize individual models with Optuna.
    5. Build stacking ensemble.
    6. Tune thresholds.
    7. Evaluate all models and log to MLflow.
    8. Save the best model.

    Parameters
    ----------
    config_dir : str
        Path to configuration directory.

    Returns
    -------
    dict
        Pipeline results including models, metrics, and comparison table.
    """
    config = load_all_configs(config_dir)
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("models", {})
    stacking_cfg = config.get("stacking", {})

    random_state = training_cfg.get("random_state", 42)
    fe_cfg = training_cfg.get("feature_engineering", {})

    # --- MLflow setup ---
    mlflow_cfg = training_cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "stroke-risk-prediction"))

    # === Step 1: Load data ===
    logger.info("Step 1: Loading data...")
    df = load_data(path=data_cfg.get("raw_path"))

    # === Step 2: Feature engineering ===
    logger.info("Step 2: Engineering features...")
    df = engineer_features(
        df,
        age_bins=fe_cfg.get("age_bins", True),
        bmi_categories=fe_cfg.get("bmi_categories", True),
        glucose_categories=fe_cfg.get("glucose_categories", True),
        interaction_features=fe_cfg.get("interaction_features", True),
        risk_score=fe_cfg.get("risk_score", True),
    )

    # === Step 3: Split data ===
    logger.info("Step 3: Splitting data...")
    target_col = data_cfg.get("target_column", "stroke")
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=target_col,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=random_state,
    )

    # === Step 4: Build preprocessor and transform ===
    logger.info("Step 4: Preprocessing...")

    # Identify new engineered categorical columns
    engineered_cat_cols = []
    base_cat = data_cfg.get("categorical_features", [])
    for col in ["age_group", "bmi_category", "glucose_category"]:
        if col in X_train.columns:
            engineered_cat_cols.append(col)

    all_cat_cols = base_cat + engineered_cat_cols

    # Numerical features include engineered numerics
    base_num = data_cfg.get("numerical_features", [])
    engineered_num_cols = [
        c
        for c in ["age_x_hypertension", "age_x_heart_disease", "bmi_x_glucose", "age_x_bmi", "risk_score"]
        if c in X_train.columns
    ]
    all_num_cols = base_num + engineered_num_cols
    bin_cols = data_cfg.get("binary_features", [])

    preprocessor = build_preprocessor(
        categorical_features=all_cat_cols,
        numerical_features=all_num_cols,
        binary_features=bin_cols,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    # === Step 5: Resample training data ===
    logger.info("Step 5: Resampling training data...")
    resampling_strategy = training_cfg.get("resampling_strategy", "smoteenn")
    X_train_resampled, y_train_resampled = resample_data(
        X_train_processed, y_train, strategy=resampling_strategy, random_state=random_state
    )

    # === Step 6: Optimize models ===
    logger.info("Step 6: Optimizing models with Optuna...")
    optimization_results = optimize_all_models(
        X=X_train_resampled,
        y=y_train_resampled,
        model_configs=model_cfg,
        scoring=training_cfg.get("scoring_metric", "f1"),
        cv_folds=training_cfg.get("cv_folds", 5),
        random_state=random_state,
    )

    # === Step 7: Build stacking ensemble ===
    logger.info("Step 7: Building stacking ensemble...")
    base_models = {name: res["best_model"] for name, res in optimization_results.items()}

    stacking_model = train_stacking_ensemble(
        base_models=base_models,
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        cv_folds=stacking_cfg.get("cv_folds", 5),
        random_state=random_state,
    )

    # Add stacking to models
    all_models = {**base_models, "stacking_ensemble": stacking_model}

    # === Step 8: Evaluate and threshold-tune ===
    logger.info("Step 8: Evaluating models and tuning thresholds...")
    threshold_cfg = training_cfg.get("threshold_tuning", {})
    threshold_method = threshold_cfg.get("method", "f1")
    recall_target = threshold_cfg.get("recall_target", 0.80)

    evaluation_results = {}

    for name, model in all_models.items():
        y_prob = model.predict_proba(X_test_processed)[:, 1]

        # Find optimal threshold
        thresh_result = find_optimal_threshold(
            y_test.values, y_prob, method=threshold_method, recall_target=recall_target
        )
        optimal_threshold = thresh_result["threshold"]

        # Predictions at optimal threshold
        y_pred = (y_prob >= optimal_threshold).astype(int)
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        evaluation_results[name] = {
            "model": model,
            "metrics": metrics,
            "threshold": optimal_threshold,
            "threshold_details": thresh_result,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        # --- Log to MLflow ---
        with mlflow.start_run(run_name=name):
            # Log params
            if name in optimization_results:
                mlflow.log_params(optimization_results[name]["best_params"])
            mlflow.log_param("resampling_strategy", resampling_strategy)
            mlflow.log_param("optimal_threshold", optimal_threshold)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Log confusion matrix
            cm_fig = plot_confusion_matrix(
                y_test.values, y_pred, title=f"Confusion Matrix -- {name}"
            )
            mlflow.log_figure(cm_fig, f"confusion_matrix_{name}.png")
            plt.close(cm_fig)

            logger.info(
                "  %s: F1=%.4f, Precision=%.4f, Recall=%.4f, ROC-AUC=%.4f (threshold=%.4f)",
                name,
                metrics["f1_stroke"],
                metrics["precision_stroke"],
                metrics["recall_stroke"],
                metrics.get("roc_auc", 0),
                optimal_threshold,
            )

    # === Step 9: Comparison and save best model ===
    logger.info("Step 9: Comparing models and saving best...")
    comparison_table = build_comparison_table(evaluation_results)
    logger.info("\n%s", comparison_table.to_string())

    # Find best model by F1 (stroke)
    best_model_name = comparison_table.iloc[0]["Model"]
    best_model = evaluation_results[best_model_name]["model"]
    best_threshold = evaluation_results[best_model_name]["threshold"]

    # Save best model and preprocessor
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    artifact = {
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "feature_engineering_config": fe_cfg,
    }
    artifact_path = models_dir / "best_model.joblib"
    joblib.dump(artifact, artifact_path)
    logger.info("Best model (%s) saved to %s", best_model_name, artifact_path)

    # Save ROC and PR curve plots
    models_for_plot = {name: res["model"] for name, res in evaluation_results.items()}
    roc_fig = plot_roc_curves(models_for_plot, X_test_processed, y_test.values)
    roc_fig.savefig(models_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(roc_fig)

    pr_fig = plot_precision_recall_curves(models_for_plot, X_test_processed, y_test.values)
    pr_fig.savefig(models_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    return {
        "models": all_models,
        "evaluation_results": evaluation_results,
        "comparison_table": comparison_table,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_threshold": best_threshold,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "X_test": X_test_processed,
        "y_test": y_test,
    }

