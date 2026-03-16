import pyautogui
import numpy as np


class Screenshoter:
    def __init__(self):
        pass

    def make_screenshot(self):
        screenshot = pyautogui.screenshot()
        image_np = np.array(screenshot)

        return image_np