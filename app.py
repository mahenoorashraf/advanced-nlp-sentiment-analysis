import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import re

from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# LOAD MODEL
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
# CREATE FASTAPI APP
# ==========================================

app = FastAPI(
    title="NLP Sentiment Analysis API",
    version="1.0"
)


# ==========================================
# REQUEST FORMAT
# ==========================================

class ReviewRequest(BaseModel):
    review: str


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")

def home():

    return {
        "message": "Sentiment Analysis API Running"
    }


# ==========================================
# PREDICT ROUTE
# ==========================================

@app.post("/predict")

def predict_sentiment(data: ReviewRequest):

    review = clean_text(data.review)

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=100,
        padding='post'
    )

    prediction = model.predict(padded)

    confidence = float(prediction[0][0])

    if confidence >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return {
        "review": data.review,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }