import streamlit as st
import pandas as pd
import joblib

# Set modern wide layout configuration
st.set_page_config(
    page_title="Thunderstorm Predictor",
    page_icon="🌦",
    layout="wide"
)

MODEL_PATH = 'model/Random_Forest_best_model.pkl'

@st.cache_resource
def load_local_model():
    """Caches the model initialization to prevent slow page reloads."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ Could not find the model file at `{MODEL_PATH}`. Please check your folder structure.")
        return None

model = load_local_model()

st.title("🌦 Thunderstorm Prediction App")
st.markdown("Enter atmospheric metrics below to predict the likelihood of real-time Convective Thunderstorm (TH) occurrences.")
st.markdown("---")

# 1. SIDEBAR DEMO PRESETS FOR QUICK TESTING
st.sidebar.header("💡 Quick-Load Test Profiles")
st.sidebar.write("Click a button below to instantly populate realistic meteorological boundaries into your dashboard:")

clear_sky_preset = {
    "sweat": 91.2, "k": -1.4, "tt": 24.7, "stability": 25.8,
    "moisture": 22.8, "convective": 0.0, "temp_press": 5636.0, "profile": 993.98
}

severe_storm_preset = {
    "sweat": 420.0, "k": 38.0, "tt": 56.0, "stability": -10.0, # Negative means massive rising instability
    "moisture": 55.0, "convective": 2500.0, "temp_press": 5700.0, "profile": 900.00
}

# Keep state persistent when user triggers a selection change
if "form_data" not in st.session_state:
    st.session_state.form_data = clear_sky_preset

if st.sidebar.button("☀️ Populate Clear Skies Profile"):
    st.session_state.form_data = clear_sky_preset

if st.sidebar.button("🚨 Populate Severe Thunderstorm Profile"):
    st.session_state.form_data = severe_storm_preset


# 2. TWO-COLUMN USER INPUT DESIGN
st.subheader("📊 Ambient Atmospheric Parameters")
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


# 3. DIRECT RUNTIME MODEL INFERENCE AND FEEDBACK BUILD
if st.button("🚀 Run Convective Thunderstorm Prediction", use_container_width=True):
    if model is not None:
        # Create input DataFrame matching your exact process pipeline sequence
        input_df = pd.DataFrame([{
            "SWEAT index": SWEAT_index,
            "K index": K_index,
            "Totals totals index": Totals_totals_index,
            "Environmental_Stability": Environmental_Stability,
            "Moisture_Indices": Moisture_Indices,
            "Convective_Potential": Convective_Potential,
            "Temperature_Pressure": Temperature_Pressure,
            "Moisture_Temperature_Profiles": Moisture_Temperature_Profiles
        }])
        
        try:
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1])
            
            st.subheader("🎯 Model Execution Analysis")
            out_col1, out_col2 = st.columns(2)
            
            with out_col1:
                if prediction == 1:
                    st.error("🚨 THUNDERSTORM DETECTED / CONVECTIVE CONDITIONS MET")
                else:
                    st.success("☀️ CLEAR WEATHER / NO CONVECTIVE THREAT")
                
                st.metric(label="Target Class Output (TH)", value=f"Class {prediction}")
                
            with out_col2:
                st.write(f"**Convective Saturation Confidence:** {probability * 100:.2f}%")
                st.progress(probability)
                st.caption("Probability threshold marker: Classification triggers class 1 above 50.00%.")
                
        except Exception as e:
            st.error("⚠️ Model Matrix Dimensions Do Not Match.")
            st.markdown(f"Your model failed execution because it expects a different number of columns. "
                        f"**Underlying System Exception:** `{str(e)}`")
