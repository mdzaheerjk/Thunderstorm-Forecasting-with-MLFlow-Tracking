from fastapi import FastAPI
from app.schemas import WeatherInput
from app.predictor import predict_weather

app = FastAPI(title='ThunderStorm Prediction API')

@app.get('/')
def home():
    return {'Message': "Weather Prediction API is operational"}

@app.post('/predict')
def predict(data: WeatherInput):
    # Keep this as a simple flat list: [val1, val2, val3...]
    features = data.to_list() 
    
    result = predict_weather(features)
    return result
