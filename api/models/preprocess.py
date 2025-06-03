import re
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(_name_)

def preprocess_text(text):
    try:
        logger.debug(f"Raw text: {text}")
        # Basic preprocessing: lowercase, remove special chars, strip
        cleaned = text.lower()
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        cleaned = cleaned.strip()
        logger.debug(f"Cleaned text: {cleaned}")
        return cleaned
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise RuntimeError(f"Preprocessing failed: {str(e)}")