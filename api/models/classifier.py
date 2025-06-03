from transformers import pipeline
from api.models.preprocess import preprocess_text
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(_name_)

class NewsClassifier:
    def _init_(self):
        try:
            self.model = pipeline(
                "text-classification",
                model="./ml/models/roberta-fake-news-new",
                tokenizer="./ml/models/roberta-fake-news-new",
                top_k=None
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def predict(self, text):
        try:
            cleaned_text = preprocess_text(text)
            logger.debug(f"Cleaned text: {cleaned_text}")
            results = self.model(cleaned_text)  # Expecting [[{label, score}, ...]]
            logger.debug(f"Raw model output: {results}")
            if not results or not isinstance(results, list) or not results[0]:
                raise ValueError("Invalid model output")
            results = results[0]  # First prediction
            labels = ["fake", "misleading", "credible"]  # LABEL_0=fake, LABEL_1=misleading, LABEL_2=credible
            max_score = max(results, key=lambda x: x["score"])
            label_str = max_score["label"]
            try:
                label_idx = int(label_str.replace("LABEL_", ""))
                if label_idx < 0 or label_idx >= len(labels):
                    raise ValueError(f"Invalid label index: {label_idx}")
            except ValueError as e:
                logger.error(f"Label parsing error: {str(e)}")
                raise ValueError(f"Invalid label format: {label_str}")
            label = labels[label_idx]
            confidence = float(max_score["score"])
            logger.debug(f"Prediction: {label}, Confidence: {confidence}")
            return label, confidence
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")