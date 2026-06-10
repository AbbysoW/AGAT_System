import pyautogui
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Screenshoter:
    def __init__(self):
        logger.debug("Screenshoter initialized")

    def make_screenshot(self):
        logger.debug("Taking screenshot")
        try:
            screenshot = pyautogui.screenshot()
            image_np = np.array(screenshot)
            logger.debug(f"Screenshot captured | shape={image_np.shape}")
            return image_np
        except Exception as e:
            logger.error(f"Screenshot error: {type(e).__name__}: {e}", exc_info=True)
            raise