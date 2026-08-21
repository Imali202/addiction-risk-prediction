import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from config import CLASSIFIER_PATH, REGRESSOR_PATH, PIPELINE_PATH, SCHEMA_PATH


@st.cache_resource
def load_artifacts():
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return classifier, regressor, pipeline, schema


def engineer_features(raw: dict) -> dict:
    """Mirrors the Colab Stage 4 engineer_features() function exactly.
    Must stay identical to the training-time version or predictions will be wrong."""
    d = dict(raw)

    d["sleep_deficit"] = 9 - d["Sleep_Hours"]

    d["checks_per_hour"] = (
        d["Phone_Checks_Per_Day"] / d["Daily_Usage_Hours"]
        if d["Daily_Usage_Hours"] > 0 else 0
    )
    d["social_to_total_ratio"] = (
        d["Time_on_Social_Media"] / d["Daily_Usage_Hours"]
        if d["Daily_Usage_Hours"] > 0 else 0
    )
    d["edu_to_total_ratio"] = (
        d["Time_on_Education"] / d["Daily_Usage_Hours"]
        if d["Daily_Usage_Hours"] > 0 else 0
    )
    d["mental_health_composite"] = (
        d["Anxiety_Level"] + d["Depression_Level"] + (10 - d["Self_Esteem"])
    ) / 3
    d["family_support_composite"] = (
        d["Family_Communication"] + d["Parental_Control"] * 10
    ) / 2
    d["bedtime_risk"] = d["Screen_Time_Before_Bed"] * max(d["sleep_deficit"], 0)
    d["weekend_escalation"] = d["Weekend_Usage_Hours"] - d["Daily_Usage_Hours"]

    return d


def build_input_dataframe(raw_form_values: dict, schema: dict) -> pd.DataFrame:
    """Takes raw user inputs, computes engineered features, and assembles
    a single-row DataFrame in the exact column order the pipeline expects."""
    full_row = engineer_features(raw_form_values)

    cols = schema["numeric_features"] + schema["categorical_features"] + schema["binary_features"]
    missing = [c for c in cols if c not in full_row]
    if missing:
        raise ValueError(f"Missing expected fields before prediction: {missing}")

    row = {col: full_row[col] for col in cols}
    return pd.DataFrame([row])


def predict_risk(raw_form_values: dict):
    classifier, regressor, pipeline, schema = load_artifacts()
    X = build_input_dataframe(raw_form_values, schema)
    X_proc = pipeline.transform(X)

    risk_tier_pred = classifier.predict(X_proc)[0]
    addiction_score = float(regressor.predict(X_proc)[0])

    return risk_tier_pred, addiction_score, X


def load_css(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)