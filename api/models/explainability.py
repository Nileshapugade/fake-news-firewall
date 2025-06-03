from api.models.preprocess import preprocess_text
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(_name_)

def explain_prediction(text, classifier):
    try:
        cleaned_text = preprocess_text(text)
        logger.debug(f"Generating explanation for: {cleaned_text}")
        # Placeholder: Return dummy explanation to avoid SHAP recursion
        tokens = cleaned_text.split()[:5]
        explanation = [{"word": token, "impact": 0.0} for token in tokens]
        logger.debug(f"Explanation: {explanation}")
        return ", ".join([f"{t['word']}({t['impact']})" for t in explanation])
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}")
        return []