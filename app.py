# Optimized Memory-Efficient Version of app.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Use smaller models for lower memory overhead
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Lazy loading function

def lazy_load_model():
    if 'lazy_model' not in globals():
        global lazy_model
        lazy_model = model
    return lazy_model

# Example usage of the lazy loading function

def predict(text):
    lazy_load_model()
    inputs = tokenizer(text, return_tensors="pt")
    outputs = lazy_model(**inputs)
    return outputs.logits

# Add your main logic here if needed