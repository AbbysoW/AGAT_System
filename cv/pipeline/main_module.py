import time
import logging

from modules.screenshoter import Screenshoter
from modules.text_recognition import TextRecognizer

logger = logging.getLogger(__name__)

class CV:
    

    def __init__(self):
        logger.info("CV init")
        try:
            self.text_recognizer = TextRecognizer()
            logger.debug("TextRecognizer initialized")
            
            self.screenshoter = Screenshoter()
            logger.debug("Screenshoter initialized")
            logger.info("CV initialized successfully")
        except Exception as e:
            logger.error(f"CV init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def analyze_screen(self):
        logger.debug("Screen analysis started")
        try:
            screenshot = self.screenshoter.make_screenshot()
            logger.debug(f"Screenshot taken | size={len(screenshot) if isinstance(screenshot, bytes) else 'unknown'}")

            text = self.text_recognizer.recognize(screenshot)
            logger.info(f"Text recognized | text_len={len(text)}")

            result = {
                "text": text,
                "objects": []
            }
            return result
        except Exception as e:
            logger.error(f"Screen analysis error: {type(e).__name__}: {e}", exc_info=True)
            raise
