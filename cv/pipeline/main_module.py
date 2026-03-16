import time

from modules.screenshoter import Screenshoter
from modules.text_recognition import TextRecognizer

class CV:
    

    def __init__(self):
        self.text_recognizer = TextRecognizer()
        self.screenshoter = Screenshoter()

    def analyze_screen(self):
        screenshot = self.screenshoter.make_screenshot()

        text = self.text_recognizer.recognize(screenshot)


        return {
            "text": text,
            "objects": []
        }
