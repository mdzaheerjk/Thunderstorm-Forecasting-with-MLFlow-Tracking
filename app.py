import streamlit as st
import pandas as pd
import joblib

MODEL_PATH='model/Random_Forest_best_model.pkl'
model=joblib.load(MODEL_PATH)

st.title("🌦 Thunderstorm Prediction App")

SWEAT_index=st.number_input('SWEAT Index')
K_index=st.number_input("K Index")
Totals_totals_index=st.number_input("Total Totals Index")
Environmental_Stability=st.number_input('Environmental Stability')
Moisture_Indices=st.number_input("Moisture Indices")
Convective_potential=st.number_input('Convective Potentail')
Temperature_Pressure=st.number_input("Temperature Pressure")
Moisture_Temperature_profiles=st.number_input("Moisture Temperature Profiles")

if st.button('Predict'):
    input_df=pd.DataFrame([{
        "SWEAT index": SWEAT_index,
        "K index": K_index,
        "Totals totals index": Totals_totals_index,
        "Environmental_Stability": Environmental_Stability,
        "Moisture_Indices": Moisture_Indices,
        "Convective_Potential": Convective_potential,
        "Temperature_Pressure": Temperature_Pressure,
        "Moisture_Temperature_Profiles": Moisture_Temperature_profiles

    }])
    prediction=model.predict(input_df)[0]
    probability=model.predict_proba(input_df)[0][1]

    st.success(f"Prediction : {prediction}")
    st.info(f"Probability : {probability:.4f}")