import streamlit as st
import pandas as pd
import joblib

# --------- Load artifacts ----------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "rf_model.joblib"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.joblib"))  # list of column names used in training

st.title("Garment Worker Productivity Prediction")
st.write("Predict **actual_productivity** using the trained Random Forest model.")

# --------- User inputs (raw/original style inputs) ----------
st.header("Input Features")

team = st.number_input("team", min_value=1, max_value=50, value=8)
targeted_productivity = st.number_input("targeted_productivity", min_value=0.0, max_value=1.5, value=0.8)
smv = st.number_input("smv", min_value=0.0, value=26.16)
wip = st.number_input("wip", min_value=0.0, value=1108.0)
over_time = st.number_input("over_time", min_value=0.0, value=7080.0)
incentive = st.number_input("incentive", min_value=0.0, value=98.0)
idle_time = st.number_input("idle_time", min_value=0.0, value=0.0)
idle_men = st.number_input("idle_men", min_value=0.0, value=0.0)
no_of_style_change = st.number_input("no_of_style_change", min_value=0, value=0)
no_of_workers = st.number_input("no_of_workers", min_value=1.0, value=59.0)

quarter = st.selectbox("quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
department = st.selectbox("department", ["finishing", "sewing", "sweing"])  # include sweing if your data had it
day = st.selectbox("day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

# --------- Build one-row dataframe in the SAME format as training features ----------
raw = {
    "team": team,
    "targeted_productivity": targeted_productivity,
    "smv": smv,
    "wip": wip,
    "over_time": over_time,
    "incentive": incentive,
    "idle_time": idle_time,
    "idle_men": idle_men,
    "no_of_style_change": no_of_style_change,
    "no_of_workers": no_of_workers,
    "quarter": quarter,
    "department": department,
    "day": day,
}

df = pd.DataFrame([raw])

# dummy encoding like your notebook
df = pd.get_dummies(df, columns=["quarter", "department", "day"], drop_first=True)

# Align columns to training features (missing -> 0)
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0

df = df[feature_cols]

# Convert True/False to 0/1 just in case
df = df.replace({True: 1, False: 0})

st.subheader("Prediction")
if st.button("Predict"):
    pred = model.predict(df)[0]
    st.success(f"Predicted actual_productivity: **{pred:.4f}**")
