from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import re
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

MAX_LEN = 300
THRESHOLD = 0.5

# =========================
# Load BiLSTM Model
# =========================
model = load_model("models/bilstm_v2/model.keras")

with open("models/bilstm_v2/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# Request Schema
# =========================
class NewsRequest(BaseModel):
    text: str

# =========================
# Preprocess
# =========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# Root
# =========================
@app.get("/")
def home():
    return {"message": "Fake News Detection API Running"}

# =========================
# Predict
# =========================
@app.post("/predict")
def predict(request: NewsRequest):

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    start = time.time()

    cleaned = preprocess(request.text)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prob = float(model.predict(padded, verbose=0)[0][0])

    prediction = "Real" if prob > THRESHOLD else "Fake"

    latency = round(time.time() - start, 4)

    return {
        "prediction": prediction,
        "confidence": round(prob, 4),
        "latency": latency
    }