from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)


print("TensorFlow model libraries imported successfully")


# ==========================================
# BUILD BIDIRECTIONAL LSTM MODEL
# ==========================================

def build_model(
    vocab_size=5000,
    embedding_dim=64,
    max_length=100
):

    print("Building model...")

    model = Sequential([

        # ==================================
        # EMBEDDING LAYER
        # ==================================

        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),

        # ==================================
        # FIRST BIDIRECTIONAL LSTM
        # ==================================

        Bidirectional(
            LSTM(
                64,
                return_sequences=True
            )
        ),

        Dropout(0.5),

        # ==================================
        # SECOND BIDIRECTIONAL LSTM
        # ==================================

        Bidirectional(
            LSTM(32)
        ),

        Dropout(0.3),

        # ==================================
        # DENSE LAYER
        # ==================================

        Dense(
            24,
            activation='relu'
        ),

        # ==================================
        # OUTPUT LAYER
        # ==================================

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

    return model


# ==========================================
# TEST MODEL
# ==========================================

if __name__ == "__main__":

    print("Starting model creation...\n")

    model = build_model()

    print("\nModel Summary:\n")

    model.summary()

    print("\nModel created successfully.")