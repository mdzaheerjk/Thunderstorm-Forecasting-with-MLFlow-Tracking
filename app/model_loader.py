import joblib 
from app.config import MODEL_PATH

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()  # Singleton loaded once
