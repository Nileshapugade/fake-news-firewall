from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

app = FastAPI()

# Load model and tokenizer explicitly for local path
model_path = "./ml/models/roberta-fake-news"
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)

class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    label: str

@app.post("/classify")
async def classify(input: TextInput):
    try:
        logger.info(f"Received text for classification: {input.text}")
        prediction = classifier(input.text)[0]
        top_prediction = max(prediction, key=lambda x: x["score"])
        return {
            "label": top_prediction["label"],
            "confidence": top_prediction["score"],
            "explanation": "Model prediction based on fine-tuned RoBERTa."
        }
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        return {"error": str(e)}

@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        logger.info(f"Received feedback: {input.dict()}")
        return {"message": "Feedback saved"}
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return {"error": str(e)}