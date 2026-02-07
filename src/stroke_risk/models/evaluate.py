"""Model evaluation -- metrics, plots, threshold tuning, and cross-validated confidence intervals."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like or None
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary of metric name -> value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_stroke": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_stroke": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_stroke": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_no_stroke": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_no_stroke": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_no_stroke": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "f1",
    recall_target: float = 0.80,
) -> dict[str, Any]:
    """Find the optimal classification threshold.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    method : str
        'f1' to maximize F1 score, or 'recall_target' to find threshold
        that achieves a minimum recall.
    recall_target : float
        Target recall (used when method='recall_target').

    Returns
    -------
    dict
        Contains 'threshold', 'f1', 'precision', 'recall'.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if method == "f1":
        # Compute F1 for each threshold
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        logger.info(
            "Optimal threshold (F1): %.4f -> F1=%.4f, Precision=%.4f, Recall=%.4f",
            best_threshold,
            f1_scores[best_idx],
            precisions[best_idx],
            recalls[best_idx],
        )

        return {
            "threshold": float(best_threshold),
            "f1": float(f1_scores[best_idx]),
            "precision": float(precisions[best_idx]),
            "recall": float(recalls[best_idx]),
        }

    elif method == "recall_target":
        # Find the highest threshold that achieves at least recall_target
        valid_mask = recalls[:-1] >= recall_target
        if not valid_mask.any():
            logger.warning(
                "Cannot achieve recall >= %.2f. Using lowest threshold.", recall_target
            )
            best_idx = 0
        else:
            # Among thresholds meeting recall target, pick highest precision
            valid_f1 = np.where(valid_mask, precisions[:-1], -1)
            best_idx = np.argmax(valid_f1)

        best_threshold = thresholds[best_idx]

        return {
            "threshold": float(best_threshold),
            "f1": float(
                2
                * precisions[best_idx]
                * recalls[best_idx]
                / (precisions[best_idx] + recalls[best_idx] + 1e-8)
            ),
            "precision": float(precisions[best_idx]),
            "recall": float(recalls[best_idx]),
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'f1' or 'recall_target'.")


def cross_validate_with_ci(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Run cross-validation and compute metrics with 95% confidence intervals.

    Parameters
    ----------
    model : estimator
        Sklearn-compatible classifier.
    X : array-like
        Features.
    y : array-like
        Labels.
    cv_folds : int
        Number of CV folds.
    scoring : list[str] or None
        Scoring metrics. Defaults to ['f1', 'precision', 'recall', 'roc_auc'].
    random_state : int
        Random seed.

    Returns
    -------
    dict
        {metric_name: {'mean': float, 'std': float, 'ci_lower': float, 'ci_upper': float}}
    """
    if scoring is None:
        scoring = ["f1", "precision", "recall", "roc_auc", "average_precision"]

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    results = {}
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_results[key]
        mean = scores.mean()
        std = scores.std()
        ci_lower = mean - 1.96 * std / np.sqrt(len(scores))
        ci_upper = mean + 1.96 * std / np.sqrt(len(scores))

        results[metric] = {
            "mean": float(mean),
            "std": float(std),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a styled confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str
        Plot title.
    save_path : str or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    labels = ["No Stroke", "Stroke"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_roc_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot ROC curves for multiple models.

    Parameters
    ----------
    models : dict
        {model_name: fitted_model}
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    save_path : str or None
        If provided, save figure.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title("ROC Curves -- Model Comparison", fontsize=14)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Precision-Recall curves for multiple models.

    Parameters
    ----------
    models : dict
        {model_name: fitted_model}
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    save_path : str or None
        If provided, save figure.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)

    # Add baseline (prevalence)
    prevalence = np.mean(y_test)
    ax.axhline(y=prevalence, color="k", linestyle="--", lw=1, label=f"Baseline ({prevalence:.3f})")
    ax.set_title("Precision-Recall Curves -- Model Comparison", fontsize=14)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def build_comparison_table(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a comparison DataFrame from model evaluation results.

    Parameters
    ----------
    results : dict
        {model_name: {'metrics': dict, 'threshold': float, ...}}

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by F1 (stroke class) descending.
    """
    rows = []
    for name, res in results.items():
        m = res.get("metrics", {})
        row = {
            "Model": name,
            "Accuracy": m.get("accuracy", np.nan),
            "Precision (Stroke)": m.get("precision_stroke", np.nan),
            "Recall (Stroke)": m.get("recall_stroke", np.nan),
            "F1 (Stroke)": m.get("f1_stroke", np.nan),
            "ROC-AUC": m.get("roc_auc", np.nan),
            "PR-AUC": m.get("pr_auc", np.nan),
            "Threshold": res.get("threshold", 0.5),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("F1 (Stroke)", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based ranking
    return df


def get_classification_report_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Return a formatted classification report string."""
    return classification_report(
        y_true,
        y_pred,
        target_names=["No Stroke", "Stroke"],
        digits=4,
    )

