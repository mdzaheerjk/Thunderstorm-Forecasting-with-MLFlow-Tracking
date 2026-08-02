import joblib 
from app.config import MODDEL_PATH

def load_model():
    with open(MODDEL_PATH,'rb') as f:
        model=joblib.load(f)
    return model

model=load_model()