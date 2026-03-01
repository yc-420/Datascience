import os
import streamlit as st
import pandas as pd
import joblib

# --------- Load artifacts ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "rf_model.joblib"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.joblib"))

st.title("Garment Worker Productivity Prediction")
st.caption("A simple deployment prototype to predict **actual_productivity** using a trained Random Forest regressor.")

with st.sidebar:
    st.header("Input Features")

    team = st.number_input("Team", min_value=1, max_value=50, value=8)
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.80, 0.01)
    smv = st.number_input("SMV", min_value=0.0, value=26.16)
    wip = st.number_input("WIP", min_value=0.0, value=1108.0)
    over_time = st.number_input("Over Time", min_value=0.0, value=7080.0)
    incentive = st.number_input("Incentive", min_value=0.0, value=98.0)
    idle_time = st.number_input("Idle Time", min_value=0.0, value=0.0)
    idle_men = st.number_input("Idle Men", min_value=0.0, value=0.0)
    no_of_style_change = st.number_input("No. of Style Change", min_value=0, value=0)
    no_of_workers = st.number_input("No. of Workers", min_value=1.0, value=59.0)

    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department", ["finishing", "sewing"])
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday","Sunday"])

    run = st.button("Predict")

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
df = pd.get_dummies(df, columns=["quarter", "department", "day"], drop_first=True)

for c in feature_cols:
    if c not in df.columns:
        df[c] = 0
df = df[feature_cols].replace({True: 1, False: 0})

st.subheader("Prediction Result")

if run:
    pred = float(model.predict(df)[0])
    st.metric("Predicted actual_productivity", f"{pred:.3f}")

st.subheader("Input Summary")
st.dataframe(pd.DataFrame([raw]))
