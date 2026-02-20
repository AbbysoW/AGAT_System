import whisper


class Whisper:

    def __init__(self, model_type: str = "small"):
        self.model = whisper.load_model(model_type)

    def get_text(self, audio):
        result = self.model.transcribe(audio)

        return result
