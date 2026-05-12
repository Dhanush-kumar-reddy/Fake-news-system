import pandas as pd

# Load ISOT dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

df_isot = pd.concat([fake, true], ignore_index=True)

# Load your new real-world training data
df_real = pd.read_csv("data/real_world_train.csv")

# Rename for consistency
df_real = df_real.rename(columns={"text": "content"})

# Combine
df_final = pd.concat([df_isot[["content", "label"]], df_real], ignore_index=True)

# Shuffle
df_final = df_final.sample(frac=1, random_state=42)

# Save
df_final.to_csv("data/final_train_v2.csv", index=False)

print("✅ Combined dataset created:", df_final.shape)