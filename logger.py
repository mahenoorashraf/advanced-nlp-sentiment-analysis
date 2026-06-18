import logging
from datetime import datetime


# ==========================================
# LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(

    filename="monitoring/app.log",

    level=logging.INFO,

    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ==========================================
# LOG PREDICTION FUNCTION
# ==========================================

def log_prediction(review, sentiment, confidence):

    logging.info(

        f"Review: {review} | "
        f"Predicted Sentiment: {sentiment} | "
        f"Confidence Score: {confidence}"
    )


# ==========================================
# LOG ERROR FUNCTION
# ==========================================

def log_error(error_message):

    logging.error(

        f"Error: {error_message}"
    )


# ==========================================
# TEST LOGGER
# ==========================================

if __name__ == "__main__":

    print("Starting monitoring logger...\n")


    # Sample prediction logs
    log_prediction(

        "This movie was amazing",

        "Positive",

        0.98
    )


    log_prediction(

        "Worst movie ever",

        "Negative",

        0.03
    )


    # Sample error log
    log_error(

        "Sample API error occurred"
    )


    print("Logs written successfully")

    print("\nCheck monitoring/app.log")