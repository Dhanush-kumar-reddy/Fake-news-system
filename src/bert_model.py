from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

def load_bert():
    tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")

    model = TFDistilBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    return tokenizer, model

def tokenize_data(tokenizer, texts, max_length=128):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )