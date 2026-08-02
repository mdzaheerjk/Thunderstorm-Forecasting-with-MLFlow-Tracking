## backend code

from fastapi import FastAPI
from app.schemas import WeatherInput
from app.predictor import predict_weather


app = FastAPI(title="Thunderstrom Prediction API")

# to ensure app is running
@app.get("/")
def home():
    return {"message": "Weather Prediction API is running"}

# microservice

@app.post("/predict")
def predict(data: WeatherInput):
    features = data.to_list()
    result = predict_weather(features) # prob , pred
    return result
