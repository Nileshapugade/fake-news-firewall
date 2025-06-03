import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(_name_)

def log_prediction(text, prediction, confidence):
    try:
        logger.debug(f"Prediction: text='{text}', prediction={prediction}, confidence={confidence}")
    except Exception as e:
        logger.error(f"Logging failed: {str(e)}")
