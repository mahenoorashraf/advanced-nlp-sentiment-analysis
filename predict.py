import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


print("Loading model and tokenizer...\n")


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = load_model("../models/final_sentiment_model.h5")

print("Model loaded successfully")


# ==========================================
# LOAD TOKENIZER
# ==========================================

with open("../data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer loaded successfully")


# ==========================================
# CLEAN TEXT FUNCTION
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_sentiment(review):

    # Clean review
    cleaned_review = clean_text(review)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_review])

    # Pad sequence
    padded = pad_sequences(
        sequence,
        maxlen=100,
        padding='post'
    )

    # Predict
    prediction = model.predict(padded)

    confidence = float(prediction[0][0])

    # Convert output
    if confidence >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, confidence


# ==========================================
# TEST PREDICTIONS
# ==========================================

sample_reviews = [

    "This movie was absolutely fantastic",

    "Worst movie ever made",

    "The acting was really good",

    "I did not like this film",

    "Amazing story and visuals"
]


print("\nRunning predictions...\n")

for review in sample_reviews:

    sentiment, confidence = predict_sentiment(review)

    print(f"Review: {review}")

    print(f"Predicted Sentiment: {sentiment}")

    print(f"Confidence Score: {confidence:.4f}")

    print("-" * 50)