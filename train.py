import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle

print("Training started...\n")

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

print("Loading processed data...")

with open("../data/processed/processed_data.pkl", "rb") as f:

    X_train, X_test, y_train, y_test = pickle.load(f)

print("Processed data loaded successfully")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ==========================================
# BUILD MODEL
# ==========================================

print("\nBuilding model...")

model = Sequential([

    Embedding(
        input_dim=5000,
        output_dim=64
    ),

    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),

    Dropout(0.5),

    Bidirectional(
        LSTM(32)
    ),

    Dropout(0.3),

    Dense(
        24,
        activation='relu'
    ),

    Dense(
        1,
        activation='sigmoid'
    )

])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully")


# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "../models/sentiment_model.h5",
    save_best_only=True
)

print("Callbacks configured")


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test, y_test),

    epochs=5,

    batch_size=32,

    callbacks=[
        early_stop,
        checkpoint
    ]
)


# ==========================================
# EVALUATE MODEL
# ==========================================

print("\nEvaluating model...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy: {accuracy:.4f}")


# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save("../models/final_sentiment_model.h5")

print("\nFinal model saved successfully")

print("\nTraining pipeline completed successfully.")