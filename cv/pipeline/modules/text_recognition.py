import pytesseract
import logging

logger = logging.getLogger(__name__)

class TextRecognizer:
    def __init__(self, tesseract_path: str = '/usr/bin/tesseract'):
        logger.debug(f"TextRecognizer init | tesseract_path={tesseract_path}")
        try:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.debug("Tesseract path configured")
        except Exception as e:
            logger.error(f"TextRecognizer init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def recognize(self, img) -> str:
        logger.debug("Text recognition started")
        try:
            text = pytesseract.image_to_string(img, lang='rus+eng')
            logger.info(f"Text recognized | text_len={len(text)}")
            return text
        except Exception as e:
            logger.error(f"Text recognition error: {type(e).__name__}: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    pass
