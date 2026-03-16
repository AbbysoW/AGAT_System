import pytesseract


class TextRecognizer:
    def __init__(self, tesseract_path: str = '/usr/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def recognize(img) -> str:
        text = pytesseract.image_to_string(img, lang='rus+eng')

        return text


if __name__ == '__main__':
    pass
