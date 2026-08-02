import pandas as pd
from app.model_loader import model

FEATURE_COLUMNS=[
    "SWEAT index",  
    "K index",
    "Totals totals index",
    "Environmental_Stability",
    "Moisture_Indices",
    "Convective_Potential",
    "Temperature_Pressure",
    "Moisture_Temperature_Profiles"

]

def predict_weather(features:list):
    df=pd.DataFrame([features],columns=FEATURE_COLUMNS)

    prediction=model.predict(df)
    proba=model.predict_proba(df)[:,1] if hasattr(model,"predict_proba") else None

    return {
        "prediction":int(prediction[0]),
        'probability':float(proba[0]) if proba is not None else None
    }