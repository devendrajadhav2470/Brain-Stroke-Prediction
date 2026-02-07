"""Pydantic schemas for the FastAPI stroke risk prediction API."""

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """Input schema for a single patient prediction request."""

    gender: str = Field(
        ...,
        description="Patient gender: 'Male' or 'Female'",
        examples=["Male"],
    )
    age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Patient age in years",
        examples=[67.0],
    )
    hypertension: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if patient has hypertension, 0 otherwise",
        examples=[0],
    )
    heart_disease: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if patient has heart disease, 0 otherwise",
        examples=[1],
    )
    ever_married: str = Field(
        ...,
        description="'Yes' or 'No'",
        examples=["Yes"],
    )
    work_type: str = Field(
        ...,
        description="Type of work: 'Private', 'Self-employed', 'Govt_job', or 'children'",
        examples=["Private"],
    )
    Residence_type: str = Field(
        ...,
        description="Residence type: 'Urban' or 'Rural'",
        examples=["Urban"],
    )
    avg_glucose_level: float = Field(
        ...,
        ge=0,
        description="Average glucose level in blood",
        examples=[228.69],
    )
    bmi: float = Field(
        ...,
        ge=0,
        le=100,
        description="Body Mass Index",
        examples=[36.6],
    )
    smoking_status: str = Field(
        ...,
        description="Smoking status: 'formerly smoked', 'never smoked', 'smokes', or 'Unknown'",
        examples=["formerly smoked"],
    )


class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution to the prediction."""

    feature: str
    shap_value: float
    abs_contribution: float


class PredictionResponse(BaseModel):
    """Output schema for a prediction response."""

    stroke_probability: float = Field(
        ...,
        description="Probability of stroke (0-1)",
    )
    risk_level: str = Field(
        ...,
        description="Risk level: 'Low', 'Medium', or 'High'",
    )
    prediction: int = Field(
        ...,
        description="Binary prediction: 1 = stroke risk, 0 = no stroke risk",
    )
    optimal_threshold: float = Field(
        ...,
        description="Classification threshold used",
    )
    top_risk_factors: list[FeatureContribution] = Field(
        ...,
        description="Top contributing features (SHAP-based explanation)",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = True
    model_name: str = ""
    version: str = "1.0.0"

