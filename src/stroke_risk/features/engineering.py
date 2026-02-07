"""Feature engineering -- domain-driven transformations for stroke risk prediction."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bin age into clinically meaningful categories.

    Categories: young (0-17), adult (18-39), middle_aged (40-59),
                senior (60-74), elderly (75+).
    """
    bins = [0, 17, 39, 59, 74, np.inf]
    labels = ["young", "adult", "middle_aged", "senior", "elderly"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


def _add_bmi_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize BMI per WHO classification.

    Categories: underweight (<18.5), normal (18.5-24.9),
                overweight (25-29.9), obese (30+).
    """
    bins = [0, 18.5, 24.9, 29.9, np.inf]
    labels = ["underweight", "normal", "overweight", "obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bins, labels=labels, right=True)
    return df


def _add_glucose_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize average glucose level based on clinical thresholds.

    Categories: normal (<100), prediabetic (100-125), diabetic (126+).
    """
    bins = [0, 100, 125, np.inf]
    labels = ["normal", "prediabetic", "diabetic"]
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"], bins=bins, labels=labels, right=True
    )
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between risk factors."""
    df["age_x_hypertension"] = df["age"] * df["hypertension"]
    df["age_x_heart_disease"] = df["age"] * df["heart_disease"]
    df["bmi_x_glucose"] = df["bmi"] * df["avg_glucose_level"]
    df["age_x_bmi"] = df["age"] * df["bmi"]
    return df


def _add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite cardiovascular risk score.

    Combines multiple risk factors into a single numeric feature.
    Higher values indicate higher stroke risk based on domain knowledge.
    """
    risk = np.zeros(len(df))

    # Age contribution (normalized 0-1)
    risk += np.clip(df["age"] / 100, 0, 1) * 3.0

    # Hypertension and heart disease are strong risk factors
    risk += df["hypertension"] * 2.0
    risk += df["heart_disease"] * 2.0

    # High glucose is a risk factor
    risk += np.where(df["avg_glucose_level"] > 125, 1.5, 0)
    risk += np.where(df["avg_glucose_level"] > 200, 1.0, 0)  # additional for very high

    # Obesity is a risk factor
    risk += np.where(df["bmi"] > 30, 1.0, 0)
    risk += np.where(df["bmi"] > 35, 0.5, 0)  # additional for severe obesity

    df["risk_score"] = risk
    return df


def engineer_features(
    df: pd.DataFrame,
    age_bins: bool = True,
    bmi_categories: bool = True,
    glucose_categories: bool = True,
    interaction_features: bool = True,
    risk_score: bool = True,
) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (will be copied, not modified in place).
    age_bins : bool
        Whether to add age group bins.
    bmi_categories : bool
        Whether to add BMI categories.
    glucose_categories : bool
        Whether to add glucose categories.
    interaction_features : bool
        Whether to add interaction features.
    risk_score : bool
        Whether to add the composite risk score.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features added.
    """
    df = df.copy()
    features_added = []

    if age_bins:
        df = _add_age_bins(df)
        features_added.append("age_group")

    if bmi_categories:
        df = _add_bmi_categories(df)
        features_added.append("bmi_category")

    if glucose_categories:
        df = _add_glucose_categories(df)
        features_added.append("glucose_category")

    if interaction_features:
        df = _add_interaction_features(df)
        features_added.extend(
            ["age_x_hypertension", "age_x_heart_disease", "bmi_x_glucose", "age_x_bmi"]
        )

    if risk_score:
        df = _add_risk_score(df)
        features_added.append("risk_score")

    logger.info("Engineered %d new features: %s", len(features_added), features_added)
    return df

