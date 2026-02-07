"""Optuna-based hyperparameter optimization for stroke risk models."""

import logging
from typing import Any

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Silence Optuna's internal logging during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_logistic_regression(trial: optuna.Trial) -> LogisticRegression:
    """Suggest hyperparameters for Logistic Regression."""
    C = trial.suggest_float("C", 1e-3, 100.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver="saga",
        max_iter=2000,
        random_state=42,
    )


def _suggest_random_forest(trial: optuna.Trial) -> RandomForestClassifier:
    """Suggest hyperparameters for Random Forest."""
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        random_state=42,
        n_jobs=-1,
    )


def _suggest_xgboost(trial: optuna.Trial) -> XGBClassifier:
    """Suggest hyperparameters for XGBoost."""
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )


def _suggest_lightgbm(trial: optuna.Trial) -> LGBMClassifier:
    """Suggest hyperparameters for LightGBM."""
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 3, 15),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


MODEL_SUGGESTORS = {
    "logistic_regression": _suggest_logistic_regression,
    "random_forest": _suggest_random_forest,
    "xgboost": _suggest_xgboost,
    "lightgbm": _suggest_lightgbm,
}


def optimize_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run Optuna hyperparameter optimization for a given model.

    Parameters
    ----------
    model_name : str
        One of: 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'.
    X : np.ndarray
        Training features (already preprocessed).
    y : np.ndarray
        Training labels.
    n_trials : int
        Number of Optuna trials.
    scoring : str
        Sklearn scoring metric to optimize (e.g., 'f1', 'average_precision').
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys: 'best_model', 'best_params', 'best_score', 'study'.
    """
    if model_name not in MODEL_SUGGESTORS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(MODEL_SUGGESTORS.keys())}"
        )

    suggestor = MODEL_SUGGESTORS[model_name]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        model = suggestor(trial)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=f"{model_name}_optimization",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Re-create best model with best params and fit on full training data
    best_trial = study.best_trial
    best_model = suggestor(best_trial)
    best_model.fit(X, y)

    logger.info(
        "Optimized %s: best %s = %.4f (trial %d/%d)",
        model_name,
        scoring,
        study.best_value,
        best_trial.number + 1,
        n_trials,
    )

    return {
        "best_model": best_model,
        "best_params": best_trial.params,
        "best_score": study.best_value,
        "study": study,
    }


def optimize_all_models(
    X: np.ndarray,
    y: np.ndarray,
    model_configs: dict[str, dict] | None = None,
    scoring: str = "f1",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Optimize all enabled models and return results.

    Parameters
    ----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    model_configs : dict or None
        Per-model config with 'enabled' and 'optuna_trials' keys.
        If None, optimizes all models with 50 trials each.
    scoring : str
        Scoring metric.
    cv_folds : int
        Cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        {model_name: optimization_result_dict} for each model.
    """
    results = {}

    for model_name in MODEL_SUGGESTORS:
        config = (model_configs or {}).get(model_name, {})
        enabled = config.get("enabled", True)
        n_trials = config.get("optuna_trials", 50)

        if not enabled:
            logger.info("Skipping %s (disabled in config).", model_name)
            continue

        logger.info("Optimizing %s with %d trials...", model_name, n_trials)
        results[model_name] = optimize_model(
            model_name=model_name,
            X=X,
            y=y,
            n_trials=n_trials,
            scoring=scoring,
            cv_folds=cv_folds,
            random_state=random_state,
        )

    return results

