import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Thunderstorm Predictor",
    page_icon="🌦",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"

st.title("🌦 Thunderstorm Prediction App")
st.markdown("---")

# 1. SIDEBAR PRESETS FOR QUICK TESTING
st.sidebar.header("💡 Quick Load Sample Data")
st.sidebar.write("Click a preset below to instantly populate realistic atmospheric metrics:")

# Define configurations
clear_sky_preset = {
    "sweat": 91.2, "k": -1.4, "tt": 24.7, "stability": 12.4,
    "moisture": 22.8, "convective": 0.0, "temp_press": 994.0, "profile": 284.4
}

severe_storm_preset = {
    "sweat": 450.0, "k": 42.0, "tt": 58.0, "stability": -7.0,
    "moisture": 60.0, "convective": 3000.0, "temp_press": 850.0, "profile": 298.7
}

# Session state initialization to hold values dynamically
if "form_data" not in st.session_state:
    st.session_state.form_data = clear_sky_preset

if st.sidebar.button("☀️ Load Clear Skies Profile"):
    st.session_state.form_data = clear_sky_preset

if st.sidebar.button("🚨 Load Severe Thunderstorm Profile"):
    st.session_state.form_data = severe_storm_preset


# 2. TWO-COLUMN USER INPUT FORM
st.subheader("📊 Input Atmospheric Parameters")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌡️ Thermal & Stability Indices")
    SWEAT_index = st.number_input('SWEAT Index', value=st.session_state.form_data["sweat"], help="Severe Weather Threat Index.")
    K_index = st.number_input("K Index", value=st.session_state.form_data["k"], help="Thunderstorm potential based on lapse rate.")
    Totals_totals_index = st.number_input("Total Totals Index", value=st.session_state.form_data["tt"], help="Static stability indicator.")
    Environmental_Stability = st.number_input('Environmental Stability', value=st.session_state.form_data["stability"], help="Negative values indicate highly unstable air.")

with col2:
    st.markdown("### 💧 Moisture & Energy Profiles")
    Moisture_Indices = st.number_input("Moisture Indices", value=st.session_state.form_data["moisture"])
    Convective_potential = st.number_input('Convective Potential (CAPE)', value=st.session_state.form_data["convective"], help="Available potential energy. Higher means more volatile.")
    Temperature_Pressure = st.number_input("Temperature Pressure", value=st.session_state.form_data["temp_press"])
    Moisture_Temperature_profiles = st.number_input("Moisture Temperature Profiles", value=st.session_state.form_data["profile"])

st.markdown("---")

# 3. PREDICTION TRIGGER AND STYLED OUTPUTS
if st.button('🚀 Run Thunderstorm Prediction', use_container_width=True):
    payload = {
        "SWEAT_index": SWEAT_index,
        "K_index": K_index,
        "Totals_totals_index": Totals_totals_index,
        "Environmental_Stability": Environmental_Stability,
        "Moisture_Indices": Moisture_Indices,
        "Convective_Potential": Convective_potential,
        "Temperature_Pressure": Temperature_Pressure,
        "Moisture_Temperature_Profiles": Moisture_Temperature_profiles
    }

    try:
        with st.spinner("Analyzing atmospheric dynamics..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            pred = result.get('prediction', 0)
            prob = result.get('probability', 0.0)

            st.subheader("🎯 Model Analysis Results")
            out_col1, out_col2 = st.columns([1, 2])

            with out_col1:
                if pred == 1:
                    st.error("🚨 THUNDERSTORM DETECTED")
                else:
                    st.success("☀️ CLEAR / NO THUNDERSTORM")
                
                st.metric(label="Binary Class Prediction", value=f"Class {pred}")

            with out_col2:
                st.write(f"**Confidence Level:** {prob * 100:.1f}%")
                st.progress(prob)
                st.caption("Probability scale: Closer to 1.0 indicates maximum convective risk.")
        else:
            st.error(f"❌ API Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to FastAPI. Make sure your Uvicorn backend server is running on port 8000!")
