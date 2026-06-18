import pandas as pd
import re
import pickle

print("Libraries imported successfully")

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("TensorFlow imported successfully")


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("../data/raw/IMDB Dataset.csv")

print("Dataset loaded successfully")
print("Dataset Shape:", df.shape)

print(df.head())


# =========================
# CLEAN TEXT FUNCTION
# =========================

def clean_text(text):

    text = str(text).lower()

    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


# =========================
# APPLY CLEANING
# =========================

df['review'] = df['review'].apply(clean_text)

print("Text cleaning completed")


# =========================
# LABEL ENCODING
# =========================

label_map = {
    'negative': 0,
    'positive': 1
}

df['sentiment'] = df['sentiment'].map(label_map)

print("Label encoding completed")


# =========================
# TOKENIZATION
# =========================

max_words = 5000
max_length = 100

tokenizer = Tokenizer(
    num_words=max_words,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df['review'])

print("Tokenizer fitted successfully")

sequences = tokenizer.texts_to_sequences(df['review'])

X = pad_sequences(
    sequences,
    maxlen=max_length,
    padding='post'
)

y = df['sentiment']

print("Tokenization and padding completed")


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-test split completed")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# =========================
# SAVE TOKENIZER
# =========================

with open("../data/processed/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved successfully")


# =========================
# SAVE PROCESSED DATA
# =========================

with open("../data/processed/processed_data.pkl", "wb") as f:
    pickle.dump(
        (X_train, X_test, y_train, y_test),
        f
    )

print("Processed data saved successfully")


# =========================
# FINAL MESSAGE
# =========================

print("\nPreprocessing completed successfully.")