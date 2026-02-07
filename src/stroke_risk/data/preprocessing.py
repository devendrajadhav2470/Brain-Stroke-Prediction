"""Data preprocessing pipelines -- encoding, scaling, splitting, and resampling."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
NUMERICAL_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
TARGET = "stroke"


def build_preprocessor(
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    binary_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for preprocessing.

    - OneHotEncoder for categorical features (handles unknown categories).
    - StandardScaler for numerical features.
    - Binary features are passed through unchanged.

    Parameters
    ----------
    categorical_features : list[str] or None
        Categorical column names. Defaults to module-level constant.
    numerical_features : list[str] or None
        Numerical column names. Defaults to module-level constant.
    binary_features : list[str] or None
        Binary column names. Defaults to module-level constant.

    Returns
    -------
    ColumnTransformer
        Fitted-ready preprocessing transformer.
    """
    cat_cols = categorical_features or CATEGORICAL_FEATURES
    num_cols = numerical_features or NUMERICAL_FEATURES
    bin_cols = binary_features or BINARY_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                cat_cols,
            ),
            (
                "bin",
                "passthrough",
                bin_cols,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A fitted ColumnTransformer.

    Returns
    -------
    list[str]
        List of output feature names.
    """
    return list(preprocessor.get_feature_names_out())


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with optional stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_column : str
        Name of the target column.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        Whether to stratify by the target column.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
    )

    logger.info(
        "Split data: train=%d, test=%d (positive rate: train=%.3f, test=%.3f)",
        len(X_train),
        len(X_test),
        y_train.mean(),
        y_test.mean(),
    )

    return X_train, X_test, y_train, y_test


def resample_data(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    strategy: str = "smoteenn",
    random_state: int = 42,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample imbalanced data using the specified strategy.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    strategy : str
        One of: 'smoteenn', 'adasyn', 'borderline_smote', 'smote_tomek', 'none'.
    random_state : int
        Random seed.
    **kwargs
        Additional keyword arguments passed to the resampler.

    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    if strategy == "none":
        logger.info("No resampling applied.")
        return X, y

    resamplers = {
        "smoteenn": lambda: SMOTEENN(
            enn=EditedNearestNeighbours(sampling_strategy="majority"),
            random_state=random_state,
            **kwargs,
        ),
        "adasyn": lambda: ADASYN(random_state=random_state, **kwargs),
        "borderline_smote": lambda: BorderlineSMOTE(
            random_state=random_state, kind="borderline-1", **kwargs
        ),
        "smote_tomek": lambda: SMOTETomek(random_state=random_state, **kwargs),
    }

    if strategy not in resamplers:
        raise ValueError(
            f"Unknown resampling strategy: {strategy}. "
            f"Choose from: {list(resamplers.keys())}"
        )

    resampler = resamplers[strategy]()
    X_res, y_res = resampler.fit_resample(X, y)

    original_pos = int(np.sum(y == 1)) if hasattr(y, "__eq__") else 0
    resampled_pos = int(np.sum(y_res == 1))

    logger.info(
        "Resampling (%s): %d -> %d samples (positive: %d -> %d)",
        strategy,
        len(y),
        len(y_res),
        original_pos,
        resampled_pos,
    )

    return X_res, y_res

