import numpy as np
import time
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

from preprocess import (
    load_data,
    preprocess_dataframe,
    split_data,
    create_tokenizer,
    text_to_sequences,
    pad_data
)

# BiLSTM Evaluation
def evaluate_bilstm():
    df = load_data()
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = split_data(df)

    tokenizer = create_tokenizer(X_train)
    X_train_seq, X_test_seq = text_to_sequences(tokenizer, X_train, X_test)
    X_train_pad, X_test_pad = pad_data(X_train_seq, X_test_seq)

    from bilstm_model import build_bilstm_model
    model = build_bilstm_model()
    model.load_weights("models/bilstm_model.h5")

    start = time.time()

    predictions = model.predict(X_test_pad)
    predictions = (predictions > 0.5).astype(int).flatten()

    latency = time.time() - start

    return y_test, predictions, latency

# BERT Evaluation
def evaluate_bert():
    df = load_data()
    df["content"] = df["title"] + " " + df["text"]

    X_train, X_test, y_train, y_test = split_data(df)

    tokenizer = DistilBertTokenizer.from_pretrained("models/bert_model/")
    model = TFDistilBertForSequenceClassification.from_pretrained("models/bert_model/")

    inputs = tokenizer(
        list(X_test),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )

    start = time.time()

    batch_size = 32
    all_preds = []

    start = time.time()

    for i in range(0, len(X_test), batch_size):
        batch_texts = list(X_test[i:i+batch_size])

        batch_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="tf"
        )

        outputs = model(batch_inputs)
        batch_preds = tf.argmax(outputs.logits, axis=1).numpy()

        all_preds.extend(batch_preds)

    latency = time.time() - start

    preds = np.array(all_preds)

    latency = time.time() - start

    return y_test, preds, latency

# Metrics Printer
def print_metrics(model_name, y_test, predictions, latency):
    print(f"\n===== {model_name} RESULTS =====")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print(f"\nLatency: {latency:.4f} seconds")

# Main Comparison
if __name__ == "__main__":
    # BiLSTM
    y_test_bi, preds_bi, lat_bi = evaluate_bilstm()
    print_metrics("BiLSTM", y_test_bi, preds_bi, lat_bi)

    # BERT
    y_test_bert, preds_bert, lat_bert = evaluate_bert()
    print_metrics("BERT", y_test_bert, preds_bert, lat_bert)

    # Final Comparison
    print("\n===== FINAL COMPARISON =====")
    print(f"BiLSTM Latency: {lat_bi:.4f} sec")
    print(f"BERT Latency:   {lat_bert:.4f} sec")