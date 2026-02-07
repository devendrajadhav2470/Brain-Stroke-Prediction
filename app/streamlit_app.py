"""Streamlit interactive demo for stroke risk prediction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model artifacts."""
    import joblib

    model_path = project_root / "models" / "best_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


artifacts = load_model()


# -------------------------------------------------------------------
# Sidebar -- patient input form
# -------------------------------------------------------------------
st.sidebar.title("Patient Information")
st.sidebar.markdown("Enter the patient's health data below.")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", min_value=0, max_value=120, value=50, step=1)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose = st.sidebar.slider("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=0.5)
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
smoking_status = st.sidebar.selectbox(
    "Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"]
)


# -------------------------------------------------------------------
# Main content
# -------------------------------------------------------------------
st.title("Stroke Risk Prediction")
st.markdown(
    """
    This application predicts the risk of brain stroke based on patient health data
    using a machine learning model trained on clinical records. The model provides
    not just a prediction, but also explains **which factors contribute most** to the
    risk assessment using SHAP (SHapley Additive exPlanations).
    """
)

if artifacts is None:
    st.error(
        "Model not found. Please run the training pipeline first:\n\n"
        "```\npython scripts/train.py\n```"
    )
    st.stop()


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
predict_button = st.sidebar.button("Predict Stroke Risk", type="primary", use_container_width=True)

if predict_button:
    from stroke_risk.features.engineering import engineer_features

    # Build input
    input_data = {
        "gender": gender,
        "age": float(age),
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    input_df = pd.DataFrame([input_data])

    # Feature engineering
    fe_config = artifacts.get("feature_engineering_config", {})
    input_df = engineer_features(
        input_df,
        age_bins=fe_config.get("age_bins", True),
        bmi_categories=fe_config.get("bmi_categories", True),
        glucose_categories=fe_config.get("glucose_categories", True),
        interaction_features=fe_config.get("interaction_features", True),
        risk_score=fe_config.get("risk_score", True),
    )

    # Preprocess and predict
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    threshold = artifacts["threshold"]
    feature_names = artifacts.get("feature_names", [])

    X_processed = preprocessor.transform(input_df)
    probability = float(model.predict_proba(X_processed)[0, 1])
    prediction = int(probability >= threshold)

    # Risk level
    if probability < 0.2:
        risk_level = "Low"
        risk_color = "#28a745"
    elif probability < 0.5:
        risk_level = "Medium"
        risk_color = "#ffc107"
    else:
        risk_level = "High"
        risk_color = "#dc3545"

    # --- Display results ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Stroke Probability", f"{probability:.1%}")

    with col2:
        st.metric("Risk Level", risk_level)

    with col3:
        st.metric("Classification Threshold", f"{threshold:.3f}")

    # --- Probability gauge ---
    st.markdown("### Risk Assessment")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Stroke Risk Score", "font": {"size": 20}},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 20], "color": "#d4edda"},
                    {"range": [20, 50], "color": "#fff3cd"},
                    {"range": [50, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- SHAP explanation ---
    st.markdown("### Feature Contributions (SHAP)")

    try:
        import shap

        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(X_processed)
            shap_vals = sv.values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
        except Exception:
            background = np.zeros((1, X_processed.shape[1]))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], background
            )
            shap_vals = explainer.shap_values(X_processed)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            shap_vals = np.atleast_2d(shap_vals)

        # Build feature contribution dataframe
        abs_vals = np.abs(shap_vals[0])
        top_k = min(10, len(feature_names))
        top_indices = np.argsort(abs_vals)[::-1][:top_k]

        contrib_data = []
        for idx in top_indices:
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            contrib_data.append(
                {
                    "Feature": fname,
                    "SHAP Value": shap_vals[0, idx],
                    "Direction": "Increases Risk" if shap_vals[0, idx] > 0 else "Decreases Risk",
                }
            )

        contrib_df = pd.DataFrame(contrib_data)

        # Horizontal bar chart
        colors = ["#dc3545" if v > 0 else "#28a745" for v in contrib_df["SHAP Value"]]

        fig_shap = go.Figure(
            go.Bar(
                x=contrib_df["SHAP Value"].values[::-1],
                y=contrib_df["Feature"].values[::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{v:+.4f}" for v in contrib_df["SHAP Value"].values[::-1]],
                textposition="outside",
            )
        )
        fig_shap.update_layout(
            title="Top Feature Contributions to Prediction",
            xaxis_title="SHAP Value (impact on stroke probability)",
            yaxis_title="",
            height=400,
            margin=dict(l=200),
        )

        st.plotly_chart(fig_shap, use_container_width=True)

        # Feature table
        st.dataframe(
            contrib_df.style.format({"SHAP Value": "{:.4f}"}),
            use_container_width=True,
        )

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    # --- Patient summary ---
    st.markdown("### Patient Summary")
    summary_df = pd.DataFrame([input_data]).T
    summary_df.columns = ["Value"]
    summary_df.index.name = "Feature"
    st.dataframe(summary_df, use_container_width=True)

else:
    st.info("Fill in the patient information in the sidebar and click **Predict Stroke Risk**.")

    # --- Model info ---
    st.markdown("### About the Model")
    st.markdown(
        f"""
        - **Model**: {artifacts.get('model_name', 'N/A')}
        - **Optimal Threshold**: {artifacts.get('threshold', 'N/A'):.4f}
        - **Features Used**: {len(artifacts.get('feature_names', []))} features
        (including engineered features)

        The model was trained using Optuna hyperparameter optimization,
        SMOTEENN resampling for class imbalance, and threshold tuning
        for optimal F1 score on the stroke class.
        """
    )

