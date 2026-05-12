from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# =====================================================
# 🔹 FASTAPI APP
# =====================================================
app = FastAPI(
    title="Fake News Detection API",
    description="Fake News Detection using BiLSTM",
    version="2.0"
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
print("🚀 Loading BiLSTM model...")

# Load trained BiLSTM v2 model
bilstm = load_model("models/bilstm_v2/model.keras")

# Load tokenizer
with open("models/bilstm_v2/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("✅ BiLSTM model loaded successfully")

# =====================================================
# 🔹 PREPROCESSING
# =====================================================
def preprocess(text: str) -> str:

    # lowercase
    text = text.lower()

    # keep only alphabets + spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =====================================================
# 🔹 PREDICTION FUNCTION
# =====================================================
def bilstm_predict(text: str):

    cleaned_text = preprocess(text)

    # Convert text -> sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    # Prediction
    prob = bilstm.predict(padded, verbose=0)[0][0]

    # Label
    prediction = "Real" if prob > THRESHOLD else "Fake"

    confidence = float(prob)

    return prediction, confidence

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
# 🔹 PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.time()

    # Input text
    text = request.text

    # ======================
    # Validation
    # ======================
    if not text.strip():

        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty"
        )

    # ======================
    # Prediction
    # ======================
    prediction, confidence = bilstm_predict(text)

    # ======================
    # Latency
    # ======================
    latency = round(time.time() - start_time, 4)

    # ======================
    # Response
    # ======================
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "model": "bilstm_v2",
        "latency": latency
    }