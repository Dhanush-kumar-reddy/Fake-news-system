import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data():
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    return df

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def preprocess_dataframe(df):
    import re

    # Case 1: ISOT dataset (has title + text)
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"] + " " + df["text"]

    # Case 2: already has content (your v2 dataset)
    elif "content" in df.columns:
        df["content"] = df["content"]

    else:
        raise ValueError("Dataset must contain either (title + text) or content column")

    # Clean text (same as before)
    df["content"] = df["content"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))

    return df[["content", "label"]]

from sklearn.model_selection import train_test_split

def split_data(df):
    return train_test_split(df["content"], df["label"], test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data()
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("Sample:", X_train.iloc[0])
    print("Label:", y_train.iloc[0])
    
    
def create_tokenizer(X_train, vocab_size=20000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    return tokenizer

def text_to_sequences(tokenizer, X_train, X_test):
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    return X_train_seq, X_test_seq

def pad_data(X_train_seq, X_test_seq, max_length=300):
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    return X_train_pad, X_test_pad

if __name__ == "__main__":
    df = load_data()
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = split_data(df)

    tokenizer = create_tokenizer(X_train)

    X_train_seq, X_test_seq = text_to_sequences(tokenizer, X_train, X_test)

    X_train_pad, X_test_pad = pad_data(X_train_seq, X_test_seq)

    print("Shape of X_train:", X_train_pad.shape)
    print("Shape of X_test:", X_test_pad.shape)