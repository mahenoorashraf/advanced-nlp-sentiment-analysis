# API Documentation

## Base URL
http://127.0.0.1:8000

## Endpoints

### GET /
Returns API status.

### POST /predict

Input:
{
    "review": "Amazing movie"
}

Output:
{
    "sentiment": "Positive",
    "confidence": 0.95
}

## API Technology
FastAPI