"""Stacking ensemble -- meta-learner on top of base model predictions."""

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def build_stacking_ensemble(
    base_models: dict[str, Any],
    meta_learner: Any | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> StackingClassifier:
    """Build a stacking ensemble from fitted base models.

    Parameters
    ----------
    base_models : dict
        {model_name: fitted_model} -- the base learners.
    meta_learner : estimator or None
        The meta-learner. Defaults to LogisticRegression.
    cv_folds : int
        Number of CV folds for generating out-of-fold predictions.
    random_state : int
        Random seed.

    Returns
    -------
    StackingClassifier
        Unfitted stacking classifier (call .fit() on training data).
    """
    if meta_learner is None:
        meta_learner = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=random_state
        )

    estimators = [(name, model) for name, model in base_models.items()]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    logger.info(
        "Built stacking ensemble with %d base learners: %s",
        len(estimators),
        [name for name, _ in estimators],
    )

    return stacking_clf


def train_stacking_ensemble(
    base_models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_learner: Any | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> StackingClassifier:
    """Build and train a stacking ensemble.

    Parameters
    ----------
    base_models : dict
        {model_name: fitted_model} -- these will be cloned internally.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    meta_learner : estimator or None
        The meta-learner.
    cv_folds : int
        CV folds for stacking.
    random_state : int
        Random seed.

    Returns
    -------
    StackingClassifier
        Fitted stacking classifier.
    """
    stacking_clf = build_stacking_ensemble(
        base_models=base_models,
        meta_learner=meta_learner,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    logger.info("Training stacking ensemble...")
    stacking_clf.fit(X_train, y_train)
    logger.info("Stacking ensemble trained successfully.")

    return stacking_clf

