"""SHAP-based model explainability for stroke risk predictions."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 500,
) -> shap.Explanation:
    """Compute SHAP values for a fitted model.

    Uses TreeExplainer for tree-based models, KernelExplainer otherwise.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible model.
    X : array-like
        Feature matrix to explain.
    feature_names : list[str] or None
        Feature names for labeling plots.
    max_samples : int
        Maximum samples to use for background data (KernelExplainer).

    Returns
    -------
    shap.Explanation
        SHAP values for the provided data.
    """
    model_type = type(model).__name__

    try:
        # Try TreeExplainer first (works for tree-based models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        logger.info("Using TreeExplainer for %s", model_type)
    except Exception:
        # Fall back to KernelExplainer
        logger.info("Falling back to KernelExplainer for %s", model_type)
        background = shap.sample(X, min(max_samples, len(X)))

        def predict_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values_raw = explainer.shap_values(X)
        shap_values = shap.Explanation(
            values=shap_values_raw,
            base_values=explainer.expected_value,
            data=X,
            feature_names=feature_names,
        )

    if feature_names is not None and hasattr(shap_values, "feature_names"):
        shap_values.feature_names = feature_names

    return shap_values


def plot_summary(
    shap_values: shap.Explanation,
    max_display: int = 15,
    save_path: str | None = None,
) -> plt.Figure:
    """Create a SHAP summary (beeswarm) plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    max_display : int
        Maximum features to display.
    save_path : str or None
        If provided, save the figure.

    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=(12, 8))

    # Handle multi-output SHAP values (take positive class)
    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[:, :, 1]

    shap.summary_plot(sv, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (Beeswarm)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to %s", save_path)

    return fig


def plot_bar(
    shap_values: shap.Explanation,
    max_display: int = 15,
    save_path: str | None = None,
) -> plt.Figure:
    """Create a SHAP bar plot (mean absolute SHAP values).

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    max_display : int
        Maximum features to display.
    save_path : str or None
        If provided, save the figure.

    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=(12, 8))

    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[:, :, 1]

    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (Bar)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP bar plot saved to %s", save_path)

    return fig


def plot_dependence(
    shap_values: shap.Explanation,
    feature: str | int,
    interaction_feature: str | int | None = "auto",
    save_path: str | None = None,
) -> plt.Figure:
    """Create a SHAP dependence plot for a specific feature.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    feature : str or int
        Feature name or index.
    interaction_feature : str, int, or None
        Feature to color by for interaction effects.
    save_path : str or None
        If provided, save the figure.

    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=(10, 6))

    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[:, :, 1]

    shap.dependence_plot(
        feature,
        sv.values,
        sv.data,
        feature_names=sv.feature_names if hasattr(sv, "feature_names") else None,
        interaction_index=interaction_feature,
        show=False,
    )
    plt.title(f"SHAP Dependence Plot -- {feature}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP dependence plot saved to %s", save_path)

    return fig


def plot_waterfall(
    shap_values: shap.Explanation,
    index: int = 0,
    max_display: int = 15,
    save_path: str | None = None,
) -> plt.Figure:
    """Create a SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    index : int
        Index of the sample to explain.
    max_display : int
        Maximum features to display.
    save_path : str or None
        If provided, save the figure.

    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=(12, 8))

    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[:, :, 1]

    shap.plots.waterfall(sv[index], max_display=max_display, show=False)
    plt.title(f"SHAP Waterfall -- Sample {index}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP waterfall plot saved to %s", save_path)

    return fig


def explain_single_prediction(
    model: Any,
    preprocessor: Any,
    input_data: dict,
    feature_names: list[str] | None = None,
) -> dict:
    """Explain a single prediction with SHAP values.

    Parameters
    ----------
    model : estimator
        Fitted model.
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    input_data : dict
        Raw input features as a dictionary.
    feature_names : list[str] or None
        Feature names after preprocessing.

    Returns
    -------
    dict
        Contains 'probability', 'prediction', 'shap_values', 'top_features'.
    """
    import pandas as pd

    input_df = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(input_df)

    probability = model.predict_proba(X_processed)[0, 1]

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_processed)
    except Exception:
        background = shap.sample(X_processed, 1)
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        raw_sv = explainer.shap_values(X_processed)
        sv = shap.Explanation(
            values=raw_sv,
            base_values=explainer.expected_value,
            data=X_processed,
            feature_names=feature_names,
        )

    # Handle multi-output
    values = sv.values
    if values.ndim == 3:
        values = values[:, :, 1]

    # Get top contributing features
    feature_labels = feature_names or [f"feature_{i}" for i in range(values.shape[1])]
    abs_contributions = np.abs(values[0])
    top_indices = np.argsort(abs_contributions)[::-1][:10]

    top_features = [
        {
            "feature": feature_labels[i],
            "shap_value": float(values[0, i]),
            "abs_contribution": float(abs_contributions[i]),
        }
        for i in top_indices
    ]

    return {
        "probability": float(probability),
        "prediction": int(probability >= 0.5),
        "shap_values": values[0].tolist(),
        "top_features": top_features,
    }

