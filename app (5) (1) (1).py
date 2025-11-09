import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Delivery Delay Prediction", page_icon="🚚")

# ---------- Model loader ----------
@st.cache_resource
def load_model():
    try:
        import joblib  # imported here so missing pkg doesn't crash import
    except Exception:
        st.error(
            "Missing dependency: **joblib**.\n\n"
            "Add `joblib` to your `requirements.txt` and redeploy."
        )
        st.stop()
    return joblib.load("decision_tree_model.pkl")

model = load_model()

# ---------- Feature schema ----------
FEATURES = [
    "Delivery_Distance",
    "Traffic_Congestion",
    "Weather_Condition",
    "Delivery_Slot",
    "Driver_Experience",
    "Num_Stops",
    "Vehicle_Age",
    "Road_Condition_Score",
    "Package_Weight",
    "Fuel_Efficiency",
    "Warehouse_Processing_Time",
]

# ---------- UI ----------
st.title("🚚 Delivery Delay Prediction")

st.write("Provide the details below to predict if a delivery delay is expected.")

col1, col2 = st.columns(2)
with col1:
    Delivery_Distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=10.0)
    Traffic_Congestion = st.number_input("Traffic Congestion (1–5)", min_value=1, max_value=5, value=3)
    Weather_Condition = st.number_input("Weather Condition (1–5)", min_value=1, max_value=5, value=3)
    Delivery_Slot = st.number_input("Delivery Slot (1-based index)", min_value=1, value=1)
    Driver_Experience = st.number_input("Driver Experience (years)", min_value=0.0, value=3.0)
    Num_Stops = st.number_input("Number of Stops", min_value=0, value=2)
with col2:
    Vehicle_Age = st.number_input("Vehicle Age (years)", min_value=0.0, value=4.0)
    Road_Condition_Score = st.number_input("Road Condition Score (1–5)", min_value=1, max_value=5, value=3)
    Package_Weight = st.number_input("Package Weight (kg)", min_value=0.0, value=5.0)
    Fuel_Efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=0.0, value=15.0)
    Warehouse_Processing_Time = st.number_input("Warehouse Processing Time (min)", min_value=0.0, value=20.0)

# Build input row in the exact feature order
row = [
    Delivery_Distance,
    Traffic_Congestion,
    Weather_Condition,
    Delivery_Slot,
    Driver_Experience,
    Num_Stops,
    Vehicle_Age,
    Road_Condition_Score,
    Package_Weight,
    Fuel_Efficiency,
    Warehouse_Processing_Time,
]

input_df = pd.DataFrame([row], columns=FEATURES).astype(float)

# ---------- Predict ----------
if st.button("Predict Delivery Delay"):
    try:
        pred = model.predict(input_df)[0]
        if int(pred) == 0:
            st.success("Predicted Delivery Delay: **0** — No significant delay expected.")
        else:
            st.warning("Predicted Delivery Delay: **1** — Delay expected.")
        st.caption("Tip: ensure your training feature order matches the inputs above.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
