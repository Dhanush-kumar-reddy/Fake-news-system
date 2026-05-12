import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess_dataframe
from bilstm_model import build_bilstm_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# 🔹 CONFIG
# =========================
MODEL_PATH = "models/bilstm_v2"
os.makedirs(MODEL_PATH, exist_ok=True)

MAX_LEN = 300
VOCAB_SIZE = 20000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# 🔹 LOAD DATA
# =========================
print("📥 Loading data...")
df = pd.read_csv("data/final_train_v2.csv")

# =========================
# 🔹 PREPROCESS (MATCH v1 EXACTLY)
# =========================
print("🧹 Preprocessing...")
df = preprocess_dataframe(df)

X = df["content"]
y = df["label"]

# =========================
# 🔹 SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y   # IMPORTANT for balance
)

# =========================
# 🔹 TOKENIZER
# =========================
print("🔤 Tokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# =========================
# 🔹 PADDING (MATCH v1)
# =========================
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

# =========================
# 🔹 MODEL
# =========================
print("🧠 Building model...")
model = build_bilstm_model()

# =========================
# 🔹 CALLBACKS (VERY IMPORTANT)
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_PATH, "best_model.keras"),
        monitor='val_loss',
        save_best_only=True
    )
]

# =========================
# 🔹 TRAIN
# =========================
print("🚀 Training started...")
history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=10,               # EarlyStopping will control actual epochs
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# =========================
# 🔹 EVALUATION
# =========================
print("\n📊 Evaluating...")

pred_probs = model.predict(X_test_pad)
preds = (pred_probs > 0.5).astype(int).flatten()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

# =========================
# 🔹 SAVE FINAL MODEL
# =========================
model.save(os.path.join(MODEL_PATH, "model.keras"))

with open(os.path.join(MODEL_PATH, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("\n✅ Model v2 saved successfully")