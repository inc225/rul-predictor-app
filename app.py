"""Streamlit app for aircraft engine RUL prediction.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_rul_model.joblib"

st.set_page_config(page_title="Aircraft Engine RUL Predictor", layout="wide")
st.title("Aircraft Engine Remaining Useful Life (RUL) Predictor")
st.caption("Predictive Maintenance for Aircraft Engines - NASA C-MAPSS FD001")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
rul_cap = bundle["rul_cap"]
best_model_name = bundle["best_model_name"]
results = bundle["results"]

st.sidebar.header("Model")
st.sidebar.write(f"Best trained model: **{best_model_name}**")
st.sidebar.dataframe(results, use_container_width=True)

st.markdown("""
This app estimates the Remaining Useful Life of a turbofan engine from operational settings and sensor readings.
You can either upload a CSV containing FD001-style rows or enter one current engine observation manually.
""")

mode = st.radio("Choose input mode", ["Upload engine data", "Manual single-cycle input"], horizontal=True)

if mode == "Upload engine data":
    uploaded = st.file_uploader("Upload a CSV with feature columns matching the training data", type=["csv"])
    st.info("For a quick demo, upload a CSV containing columns such as setting_1, setting_2, sensor_2, sensor_3, etc. The app will use the last row for each engine_id if engine_id exists.")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "engine_id" in df.columns and "cycle" in df.columns:
            df = df.loc[df.groupby("engine_id")["cycle"].idxmax()].sort_values("engine_id")
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            pred = np.clip(model.predict(df[feature_cols]), 0, rul_cap)
            out = df.copy()
            out["predicted_RUL"] = pred.round(2)
            st.success("Prediction complete")
            st.dataframe(out, use_container_width=True)
else:
    st.subheader("Manual input")
    st.write("Default values are populated from typical FD001 operating ranges. Adjust them and click Predict.")
    cols = st.columns(3)
    values = {}
    defaults = {
        "setting_1": 0.0, "setting_2": 0.0,
        "sensor_2": 642.5, "sensor_3": 1589.0, "sensor_4": 1405.0,
        "sensor_7": 553.5, "sensor_8": 2388.1, "sensor_9": 9050.0,
        "sensor_11": 47.5, "sensor_12": 521.7, "sensor_13": 2388.1,
        "sensor_14": 8130.0, "sensor_15": 8.43, "sensor_17": 392.0,
        "sensor_20": 38.8, "sensor_21": 23.3,
    }
    for idx, col in enumerate(feature_cols):
        with cols[idx % 3]:
            values[col] = st.number_input(col, value=float(defaults.get(col, 0.0)), format="%.5f")
    if st.button("Predict RUL", type="primary"):
        X = pd.DataFrame([values])[feature_cols]
        pred = float(np.clip(model.predict(X)[0], 0, rul_cap))
        st.metric("Predicted Remaining Useful Life", f"{pred:.1f} cycles")
        if pred < 30:
            st.warning("High maintenance priority: predicted RUL is below 30 cycles.")
        elif pred < 70:
            st.info("Medium maintenance priority: monitor this engine closely.")
        else:
            st.success("Low immediate risk based on the current model prediction.")
