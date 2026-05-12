import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification
)

# =========================================
# 🔹 CONFIG
# =========================================
MODEL_PATH = "models/bert_v2"

os.makedirs(MODEL_PATH, exist_ok=True)

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 3e-5

# =========================================
# 🔹 LOAD DATA
# =========================================
print("📥 Loading dataset...")

df = pd.read_csv("data/final_train_v2.csv")

# Remove null rows
df = df.dropna(subset=["content", "label"])

# Ensure correct types
df["content"] = df["content"].astype(str)
df["label"] = df["label"].astype(int)

print(f"✅ Total Samples: {len(df)}")

# =========================================
# 🔹 TRAIN / TEST SPLIT
# =========================================
print("✂️ Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================================
# 🔹 LOAD TOKENIZER + MODEL
# =========================================
print("🤖 Loading DistilBERT...")

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# =========================================
# 🔹 TOKENIZATION
# =========================================
print("🔤 Tokenizing data...")

train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="tf"
)

test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="tf"
)

# Convert labels
y_train = np.array(y_train)
y_test = np.array(y_test)

# =========================================
# 🔹 COMPILE MODEL
# =========================================
print("⚙️ Compiling model...")

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)

# =========================================
# 🔹 TRAIN
# =========================================
print("🚀 Training started...")

history = model.fit(
    dict(train_encodings),
    y_train,
    validation_data=(dict(test_encodings), y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================================
# 🔹 EVALUATION
# =========================================
print("\n📊 Evaluating model...")

outputs = model.predict(dict(test_encodings))

logits = outputs.logits

predictions = tf.argmax(logits, axis=1).numpy()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# =========================================
# 🔹 SAVE MODEL
# =========================================
print("\n💾 Saving model...")

model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print(f"\n✅ BERT v2 saved successfully at: {MODEL_PATH}")