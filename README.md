# Advanced NLP Sentiment Analysis

## 📌 Project Overview

This project performs sentiment analysis on movie reviews using Natural Language Processing (NLP) and Deep Learning techniques. It classifies reviews into **Positive** and **Negative** sentiments and provides predictions through a FastAPI REST API.

---

## 🚀 Features

* Text preprocessing and cleaning
* TF-IDF feature extraction
* Logistic Regression model
* Bidirectional LSTM model
* Model evaluation with accuracy and classification report
* FastAPI REST API
* Prediction pipeline
* Logging and monitoring
* Docker support
* Jupyter notebooks for experimentation

---

## 📂 Project Structure

```
advanced-nlp-sentiment-analysis
│
├── api/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── deployment/
├── docs/
├── models/
│   └── lstm_model.py
├── monitoring/
│   ├── logger.py
│   ├── metrics.py
│   └── app.log
├── notebooks/
│   └── model.ipynb
├── tests/
├── training/
│   └── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

## 📊 Dataset

* Dataset: IMDB Movie Reviews Dataset
* Total Reviews: 50,000
* Classes:

  * Positive
  * Negative

---

## 🧠 Models Used

### Logistic Regression

* TF-IDF Vectorization
* Binary Classification

### Bidirectional LSTM

Architecture:

```
Embedding Layer
↓
Bidirectional LSTM (128)
↓
Dropout
↓
Bidirectional LSTM (64)
↓
Dropout
↓
Dense Layer
↓
Sigmoid Output Layer
```

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/advanced-nlp-sentiment-analysis.git
cd advanced-nlp-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

Run:

```bash
python training/train.py
```

---

## 🔮 Prediction

Run:

```bash
python predict.py
```

Example Output:

```
Review: This movie was fantastic
Predicted Sentiment: Positive
Confidence Score: 0.935
```

---

## 🌐 FastAPI Deployment

Start API server:

```bash
uvicorn api.app:app --reload
```

Open:

Swagger UI:

```
http://127.0.0.1:8000/docs
```

Example Request:

```json
{
  "review": "This movie was fantastic"
}
```

Example Response:

```json
{
  "review": "This movie was fantastic",
  "sentiment": "Positive",
  "confidence": 0.935
}
```

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Test Accuracy:

```
85%
```

---

## 📋 Monitoring

Files:

* `monitoring/logger.py`
* `monitoring/metrics.py`

Logs are stored in:

```
monitoring/app.log
```

---

## 🐳 Docker Support

Build image:

```bash
docker build -t sentiment-analysis .
```

Run container:

```bash
docker run -p 8000:8000 sentiment-analysis
```

---

## 📚 Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* FastAPI
* Uvicorn
* Docker
* Jupyter Notebook

---

## 📌 Future Improvements

* Transformer models (BERT)
* Multi-class sentiment analysis
* Cloud deployment
* Kubernetes support
* Real-time monitoring dashboard

---

## 👨‍💻 Author

Mahenoor Ashraf

---

## 📄 License

This project is developed for educational and internship purposes.
