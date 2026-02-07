# Stroke Risk ML

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end machine learning pipeline for **brain stroke risk prediction** with Optuna hyperparameter optimization, SHAP explainability, MLflow experiment tracking, and deployment via FastAPI + Streamlit.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Deployment](#deployment)
- [Methodology](#methodology)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## Overview

Stroke is the **2nd leading cause of death** globally (WHO). This project builds a production-ready ML system that:

1. **Predicts stroke risk** from 10 clinical features (age, BMI, glucose level, hypertension, etc.)
2. **Explains predictions** using SHAP -- critical for healthcare ML adoption
3. **Optimizes for the right metric** -- F1 score on the minority (stroke) class, not accuracy
4. **Serves predictions** via a REST API and interactive web app

### Key Challenges Addressed

- **Severe class imbalance** (~95% no-stroke, ~5% stroke) -- handled with SMOTEENN resampling and threshold tuning
- **Hyperparameter optimization** -- Optuna with TPE sampler replaces naive grid search
- **Model interpretability** -- SHAP provides both global and per-patient explanations
- **Reproducibility** -- YAML configs, MLflow tracking, fixed random seeds

---

## Architecture

```
Raw CSV ──> Feature Engineering ──> Preprocessing ──> Resampling
                                                         │
                                                         ▼
                                               Optuna HPO (4 models)
                                                         │
                                                         ▼
                                               Stacking Ensemble
                                                         │
                                                         ▼
                                               Threshold Tuning
                                                         │
                                                    ┌────┴────┐
                                                    ▼         ▼
                                              MLflow Log   Best Model
                                                              │
                                                    ┌─────────┼──────────┐
                                                    ▼         ▼          ▼
                                              FastAPI    Streamlit    SHAP
                                              (API)      (Demo)    (Explain)
```

---

## Project Structure

```
stroke-risk-ml/
├── configs/                  # YAML configuration files
│   ├── data.yaml             #   data paths, feature lists, split ratios
│   ├── model.yaml            #   model hyperparameters, Optuna search spaces
│   └── training.yaml         #   training settings, resampling config
├── src/stroke_risk/          # Core Python package
│   ├── data/                 #   data loading & preprocessing pipelines
│   ├── features/             #   feature engineering (age bins, risk score, etc.)
│   ├── models/               #   training, Optuna optimization, stacking, evaluation
│   ├── explainability/       #   SHAP analysis (global + local explanations)
│   └── utils/                #   config loader
├── notebooks/                # Polished Jupyter notebooks
│   ├── 01_eda_and_story.ipynb        # narrative EDA with insights
│   └── 02_modeling_and_results.ipynb # full pipeline walkthrough
├── api/                      # FastAPI REST API
│   ├── main.py               #   /predict and /health endpoints
│   └── schemas.py            #   Pydantic request/response models
├── app/
│   └── streamlit_app.py      # Interactive Streamlit demo
├── scripts/
│   └── train.py              # CLI training entry point
├── models/                   # Saved model artifacts (gitignored)
├── Dockerfile                # Container image
├── docker-compose.yaml       # Multi-service deployment
├── pyproject.toml            # Package configuration
└── requirements.txt          # Pinned dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/devendrajadhav2470/stroke-risk-ml.git
cd stroke-risk-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Training Pipeline

Run the full pipeline (data loading, feature engineering, Optuna HPO, stacking, threshold tuning, MLflow logging):

```bash
python scripts/train.py
```

This will:
1. Download the dataset (cached locally)
2. Engineer 8 new features (age bins, BMI categories, interaction features, risk score)
3. Optimize 4 models with Optuna (LR, RF, XGBoost, LightGBM)
4. Build a stacking ensemble
5. Tune classification thresholds via PR-curve analysis
6. Log all experiments to MLflow
7. Save the best model to `models/best_model.joblib`

### View MLflow Dashboard

```bash
mlflow ui --backend-store-uri mlruns
```

Then open http://localhost:5000.

---

## Deployment

### Option 1: FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

- **Predict**: `POST http://localhost:8000/predict`
- **Health**: `GET http://localhost:8000/health`
- **Docs**: http://localhost:8000/docs (Swagger UI)

Example request:
```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 1,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

### Option 2: Streamlit

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 for the interactive demo.

### Option 3: Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- App: http://localhost:8501

---

## Methodology

### Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| `age_group` | Categorical | Young / Adult / Middle-aged / Senior / Elderly |
| `bmi_category` | Categorical | WHO BMI classification |
| `glucose_category` | Categorical | Normal / Prediabetic / Diabetic |
| `age_x_hypertension` | Numerical | Age interaction with hypertension |
| `age_x_heart_disease` | Numerical | Age interaction with heart disease |
| `bmi_x_glucose` | Numerical | BMI-glucose interaction |
| `age_x_bmi` | Numerical | Age-BMI interaction |
| `risk_score` | Numerical | Composite cardiovascular risk score |

### Preprocessing

- **Numerical**: StandardScaler
- **Categorical**: OneHotEncoder (drop="if_binary")
- **Binary**: Passthrough
- **Pipeline**: sklearn ColumnTransformer (no data leakage)

### Class Imbalance

- SMOTEENN (SMOTE oversampling + Edited Nearest Neighbours cleaning)
- Threshold tuning via precision-recall curve optimization

### Models Optimized

| Model | HPO Trials | Optimizer |
|-------|-----------|-----------|
| Logistic Regression | 50 | Optuna TPE |
| Random Forest | 60 | Optuna TPE |
| XGBoost | 80 | Optuna TPE |
| LightGBM | 80 | Optuna TPE |
| Stacking Ensemble | -- | Meta-learner (LR) on out-of-fold predictions |

### Evaluation Metrics

- **Primary**: F1 Score (Stroke class), PR-AUC
- **Secondary**: ROC-AUC, Accuracy, Precision, Recall
- Cross-validated with 95% confidence intervals

---

## Results

> **Note**: Run `python scripts/train.py` to generate results. The table below is a template.

| Model | F1 (Stroke) | Precision | Recall | ROC-AUC | PR-AUC | Threshold |
|-------|------------|-----------|--------|---------|--------|-----------|
| Stacking Ensemble | -- | -- | -- | -- | -- | -- |
| LightGBM | -- | -- | -- | -- | -- | -- |
| XGBoost | -- | -- | -- | -- | -- | -- |
| Random Forest | -- | -- | -- | -- | -- | -- |
| Logistic Regression | -- | -- | -- | -- | -- | -- |

*Results will be populated after training. View detailed metrics in MLflow.*

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| ML Framework | scikit-learn, XGBoost, LightGBM |
| HPO | Optuna (TPE sampler) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |
| Configuration | YAML |
| Data Handling | pandas, imbalanced-learn |

---

## License

This project is licensed under the MIT License.

---

*Built by [Devendra Bhaginath Jadhav](https://github.com/devendrajadhav2470)*
