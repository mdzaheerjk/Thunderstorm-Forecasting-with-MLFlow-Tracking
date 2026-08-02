import streamlit as st
import requests

# 1. PAGE SETUP
st.set_page_config(
    page_title="Thunderstorm Predictor",
    page_icon="🌦",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"

st.title("🌦 Weather Thunderstorm Prediction App")
st.markdown("Enter atmospheric parameters to predict TH (Thunderstorm Occurrence)")
st.markdown("---")


# 2. SIDEBAR DEMO PRESETS FOR QUICK TESTING
st.sidebar.header("💡 Quick Load Sample Data")
st.sidebar.write("Click a preset below to instantly populate realistic atmospheric metrics:")

# Meteorological baseline configurations
clear_sky_preset = {
    "sweat": 91.2, "k": -1.4, "tt": 24.7, "stability": 25.8,
    "moisture": 22.8, "convective": 0.0, "temp_press": 5636.0, "profile": 993.98
}

severe_storm_preset = {
    "sweat": 420.0, "k": 38.0, "tt": 56.0, "stability": -10.0,  # Negative means massive rising instability
    "moisture": 55.0, "convective": 2500.0, "temp_press": 5700.0, "profile": 900.0
}

# Track application state persistently across clicks
if "form_data" not in st.session_state:
    st.session_state.form_data = clear_sky_preset

if st.sidebar.button("☀️ Load Clear Skies Profile"):
    st.session_state.form_data = clear_sky_preset

if st.sidebar.button("🚨 Load Severe Thunderstorm Profile"):
    st.session_state.form_data = severe_storm_preset


# 3. TWO-COLUMN LAYOUT FOR USER INPUTS
st.subheader("📊 Atmospheric Metrics")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌡️ Thermal & Stability Indices")
    SWEAT_index = st.number_input("SWEAT Index", value=st.session_state.form_data["sweat"], help="Severe Weather Threat Index baseline.")
    K_index = st.number_input("K Index", value=st.session_state.form_data["k"], help="Vertical temperature lapse tracking point.")
    Totals_totals_index = st.number_input("Totals Totals Index", value=st.session_state.form_data["tt"], help="Static stability framework element.")
    Environmental_Stability = st.number_input("Environmental Stability", value=st.session_state.form_data["stability"], help="Calculated using Showalter + Lifted. Highly negative values imply extreme updraft potential.")

with col2:
    st.markdown("### 💧 Moisture & Geometric Profiles")
    Moisture_Indices = st.number_input("Moisture Indices", value=st.session_state.form_data["moisture"], help="Precipitable water depth saturation calculation.")
    Convective_Potential = st.number_input("Convective Potential", value=st.session_state.form_data["convective"], help="Calculated using CAPE + CINE energy thresholds.")
    Temperature_Pressure = st.number_input("Temperature Pressure", value=st.session_state.form_data["temp_press"], help="1000-500 hPa Thickness index framework metric.")
    Moisture_Temperature_Profiles = st.number_input("Moisture Temperature Profiles", value=st.session_state.form_data["profile"], help="Pressure at Lifted Condensation Level (PLCL).")

st.markdown("---")


# 4. PREDICTION TRIGGER AND STYLED FEEDBACK
if st.button("🚀 Run Thunderstorm Prediction", use_container_width=True):
    # Keys matched precisely to Pydantic requirements
    payload = {
        "SWEAT_index": SWEAT_index,
        "K_index": K_index,
        "Totals_totals_index": Totals_totals_index,
        "Environmental_Stability": Environmental_Stability,
        "Moisture_Indices": Moisture_Indices,
        "Convective_Potential": Convective_Potential,
        "Temperature_Pressure": Temperature_Pressure,
        "Moisture_Temperature_Profiles": Moisture_Temperature_Profiles,
    }

    try:
        with st.spinner("Analyzing atmospheric dynamics..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            pred = result.get('prediction', 0)
            prob = result.get('probability', 0.0)

            # Check if backend threw an internal training shape error log
            if "error_log" in result:
                st.error("⚠️ Model Matrix Dimensions Do Not Match training data shape.")
                st.info(f"Backend message: {result['error_log']}")
            else:
                st.subheader("🎯 Model Analysis Results")
                out_col1, out_col2 = st.columns(2)

                with out_col1:
                    if pred == 1:
                        st.error("🚨 THUNDERSTORM DETECTED / CONVECTIVE CONDITIONS MET")
                    else:
                        st.success("☀️ CLEAR WEATHER / NO CONVECTIVE THREAT")
                    
                    st.metric(label="Target Class Output (TH)", value=f"Class {pred}")

                with out_col2:
                    st.write(f"**Convective Saturation Confidence:** {prob * 100:.2f}%")
                    st.progress(prob)
                    st.caption("Probability threshold marker: Classification triggers class 1 above 50.00%.")
        else:
            st.error(f"❌ API Error {response.status_code}: Please verify that your FastAPI backend service is accepting requests.")
            
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection Failure: Could not reach FastAPI. Make sure your Uvicorn backend server is running on port 8000!")
