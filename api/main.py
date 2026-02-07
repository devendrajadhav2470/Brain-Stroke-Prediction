"""FastAPI application for stroke risk prediction."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from api.schemas import (
    FeatureContribution,
    HealthResponse,
    PatientInput,
    PredictionResponse,
)
from stroke_risk.features.engineering import engineer_features

logger = logging.getLogger(__name__)

# Global model artifacts
_artifacts: dict[str, Any] = {}


def _load_model_artifacts() -> dict[str, Any]:
    """Load saved model artifacts from disk."""
    model_path = Path("models/best_model.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found at {model_path}. "
            "Run 'python scripts/train.py' first."
        )

    artifacts = joblib.load(model_path)
    logger.info(
        "Loaded model artifacts: model=%s, threshold=%.4f",
        artifacts["model_name"],
        artifacts["threshold"],
    )
    return artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _artifacts
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )
    try:
        _artifacts = _load_model_artifacts()
        logger.info("Model loaded successfully on startup.")
    except FileNotFoundError as e:
        logger.warning("Model not found: %s", e)
    yield
    _artifacts.clear()


app = FastAPI(
    title="Stroke Risk Prediction API",
    description=(
        "Predicts brain stroke risk based on patient health data. "
        "Uses an ML model with SHAP-based explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _prepare_input(patient: PatientInput) -> pd.DataFrame:
    """Convert patient input to a feature-engineered DataFrame."""
    input_dict = patient.model_dump()
    df = pd.DataFrame([input_dict])

    # Apply feature engineering
    fe_config = _artifacts.get("feature_engineering_config", {})
    df = engineer_features(
        df,
        age_bins=fe_config.get("age_bins", True),
        bmi_categories=fe_config.get("bmi_categories", True),
        glucose_categories=fe_config.get("glucose_categories", True),
        interaction_features=fe_config.get("interaction_features", True),
        risk_score=fe_config.get("risk_score", True),
    )

    return df


def _get_risk_level(probability: float) -> str:
    """Map probability to a risk level."""
    if probability < 0.2:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    else:
        return "High"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = "model" in _artifacts
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_name=_artifacts.get("model_name", "not_loaded"),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """Predict stroke risk for a patient.

    Returns probability, risk level, binary prediction, and
    SHAP-based feature explanations.
    """
    if "model" not in _artifacts:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first.",
        )

    try:
        model = _artifacts["model"]
        preprocessor = _artifacts["preprocessor"]
        threshold = _artifacts["threshold"]
        feature_names = _artifacts.get("feature_names", [])

        # Prepare and preprocess input
        df = _prepare_input(patient)
        X_processed = preprocessor.transform(df)

        # Get probability
        probability = float(model.predict_proba(X_processed)[0, 1])
        prediction = int(probability >= threshold)
        risk_level = _get_risk_level(probability)

        # SHAP explanation
        top_features = []
        try:
            import shap

            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer(X_processed)
                values = sv.values
                if values.ndim == 3:
                    values = values[:, :, 1]
            except Exception:
                background = np.zeros((1, X_processed.shape[1]))
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1], background
                )
                values = explainer.shap_values(X_processed)
                if isinstance(values, list):
                    values = values[1] if len(values) > 1 else values[0]
                values = np.atleast_2d(values)

            abs_vals = np.abs(values[0])
            top_indices = np.argsort(abs_vals)[::-1][:5]

            for idx in top_indices:
                fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                top_features.append(
                    FeatureContribution(
                        feature=fname,
                        shap_value=float(values[0, idx]),
                        abs_contribution=float(abs_vals[idx]),
                    )
                )
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)

        return PredictionResponse(
            stroke_probability=round(probability, 4),
            risk_level=risk_level,
            prediction=prediction,
            optimal_threshold=round(threshold, 4),
            top_risk_factors=top_features,
        )

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

