from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# =====================================================
# 🔹 FASTAPI APP
# =====================================================
app = FastAPI(
    title="Fake News Detection API",
    description="Fake News Detection using BiLSTM v2",
    version="1.0"
)

# =====================================================
# 🔹 CONFIG
# =====================================================
MAX_LEN = 300
THRESHOLD = 0.5

# =====================================================
# 🔹 REQUEST SCHEMA
# =====================================================
class PredictionRequest(BaseModel):
    text: str

# =====================================================
# 🔹 LOAD MODEL
# =====================================================
print("🚀 Loading BiLSTM v2 model...")

model = load_model("models/bilstm_v2/model.keras")

with open("models/bilstm_v2/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("✅ BiLSTM model loaded successfully")

# =====================================================
# 🔹 PREPROCESSING
# =====================================================
def preprocess(text: str) -> str:

    text = text.lower()

    # keep alphabets only
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =====================================================
# 🔹 PREDICTION FUNCTION
# =====================================================
def predict_news(text: str):

    cleaned_text = preprocess(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    prob = model.predict(padded, verbose=0)[0][0]

    prediction = "Real" if prob > THRESHOLD else "Fake"

    return prediction, float(prob)

# =====================================================
# 🔹 ROOT ENDPOINT
# =====================================================
@app.get("/")
def home():

    return {
        "message": "Fake News Detection API",
        "model": "BiLSTM v2",
        "status": "running"
    }

# =====================================================
# 🔹 HEALTH CHECK
# =====================================================
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# =====================================================
# 🔹 PREDICT ENDPOINT
# =====================================================
@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.time()

    text = request.text

    # ==========================
    # INPUT VALIDATION
    # ==========================
    if not text.strip():

        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty"
        )

    # ==========================
    # PREDICTION
    # ==========================
    prediction, confidence = predict_news(text)

    latency = round(time.time() - start_time, 4)

    # ==========================
    # RESPONSE
    # ==========================
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "latency": latency,
        "model": "bilstm_v2"
    }