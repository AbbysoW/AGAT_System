import whisper


class Whisper:

    def __init__(self, model_type: str = "small"):
        self.model = whisper.load_model(model_type, device="cuda")

    def get_text(self, audio):
        result = self.model.transcribe(
            audio, 
            fp16=True,
            temperature=0.,
            beam_size=1,
            best_of=1,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False
            )

        return result
