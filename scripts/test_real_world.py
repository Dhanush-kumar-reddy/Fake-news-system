import sys
import os

sys.path.append(os.path.abspath("."))

import pandas as pd
from api.main import bilstm_predict
from api.main import bert_predict

# =========================
# 🔹 LOAD TEST DATA
# =========================
df = pd.read_csv("data/real_world_test.csv")

print(f"\n📊 Total Samples: {len(df)}")

correct = 0

real_correct = 0
fake_correct = 0

real_total = 0
fake_total = 0

# =========================
# 🔹 TEST LOOP
# =========================
for i, row in df.iterrows():

    text = row["text"]
    true_label = row["label"]
    source = row["source"]

    prediction, confidence = bert_predict(text)

    pred_label = 1 if prediction == "Real" else 0

    # Accuracy count
    if pred_label == true_label:
        correct += 1

    # Per-class tracking
    if true_label == 1:
        real_total += 1
        if pred_label == 1:
            real_correct += 1
    else:
        fake_total += 1
        if pred_label == 0:
            fake_correct += 1

    # =========================
    # 🔹 PRINT RESULT
    # =========================
    print("\n" + "="*60)

    print(f"Sample: {i+1}")
    print(f"Source: {source}")

    print(f"True Label : {'Real' if true_label == 1 else 'Fake'}")
    print(f"Prediction : {prediction}")

    print(f"Confidence : {confidence:.4f}")

    # Failure highlight
    if pred_label != true_label:
        print("❌ WRONG PREDICTION")
    else:
        print("✅ CORRECT")

# =========================
# 🔹 FINAL METRICS
# =========================
overall_acc = correct / len(df)

real_acc = real_correct / real_total if real_total > 0 else 0
fake_acc = fake_correct / fake_total if fake_total > 0 else 0

print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)

print(f"Overall Accuracy : {overall_acc:.4f}")

print(f"Real Accuracy    : {real_acc:.4f}")
print(f"Fake Accuracy    : {fake_acc:.4f}")

print(f"\nTotal Correct    : {correct}")
print(f"Total Samples    : {len(df)}")