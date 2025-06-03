from fastapi import APIRouter, HTTPException
from api.schemas.input import NewsInput
from api.models.classifier import NewsClassifier
from api.models.explainability import explain_prediction
from api.utils.logger import log_prediction
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(_name_)

router = APIRouter()
classifier = NewsClassifier()

@router.post("/classify")
async def classify_news(input: NewsInput):
    try:
        label, confidence = classifier.predict(input.text)
        explanation = explain_prediction(input.text, classifier)

        # Optional: log prediction somewhere
        log_prediction(input.text, label, confidence)

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),  # percentage
            "explanation": explanation  # structured list of tokens
        }
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification failed")

