import os
import pickle
import numpy as np
import tensorflow as tf

from preprocess import (
    load_data,
    preprocess_dataframe,
    split_data,
    create_tokenizer,
    text_to_sequences,
    pad_data
)

from bilstm_model import build_bilstm_model
from bert_model import load_bert, tokenize_data


# =========================
# 🔹 Config (Versioning)
# =========================
BILSTM_PATH = "models/bilstm_v1"
BERT_PATH = "models/bert_v1"

os.makedirs(BILSTM_PATH, exist_ok=True)
os.makedirs(BERT_PATH, exist_ok=True)


# =========================
# 🔹 BiLSTM Training
# =========================
def train_bilstm():
    print("\n🚀 Training BiLSTM...")

    df = load_data()
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # Tokenizer (VERY IMPORTANT)
    tokenizer = create_tokenizer(X_train)

    X_train_seq, X_test_seq = text_to_sequences(tokenizer, X_train, X_test)
    X_train_pad, X_test_pad = pad_data(X_train_seq, X_test_seq)

    model = build_bilstm_model()

    model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_test_pad, y_test),
        epochs=5,
        batch_size=64
    )

    # 🔥 Save model weights
    model.save(os.path.join(BILSTM_PATH, "model.keras"))

    # 🔥 Save tokenizer (CRITICAL)
    with open(os.path.join(BILSTM_PATH, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ BiLSTM training completed & saved")


# =========================
# 🔹 BERT Training
# =========================
def train_bert():
    print("\n🚀 Training BERT...")

    df = load_data()

    # NO aggressive cleaning
    df["content"] = df["title"] + " " + df["text"]

    # Reduce dataset for speed
    df = df.sample(n=10000, random_state=42)

    X_train, X_test, y_train, y_test = split_data(df)

    tokenizer, model = load_bert()

    train_encodings = tokenize_data(tokenizer, X_train)
    test_encodings = tokenize_data(tokenizer, X_test)

    # Convert BatchEncoding → dict
    train_encodings = dict(train_encodings)
    test_encodings = dict(test_encodings)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=3e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit(
        train_encodings,
        y_train,
        validation_data=(test_encodings, y_test),
        epochs=1,
        batch_size=8
    )

    # Save model + tokenizer
    model.save_pretrained(BERT_PATH)
    tokenizer.save_pretrained(BERT_PATH)

    print("✅ BERT training completed & saved")


# =========================
# 🔹 Main Entry
# =========================
if __name__ == "__main__":
    train_bilstm()
    train_bert()