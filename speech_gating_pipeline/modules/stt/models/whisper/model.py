import whisper
import logging

logger = logging.getLogger(__name__)

class Model:

    def __init__(self, model_type: str = "small"):
        logger.info(f"Loading Whisper model | type={model_type}")
        try:
            self.model = whisper.load_model(model_type, device="cpu")
            logger.info(f"Whisper model loaded | type={model_type}")
        except Exception as e:
            logger.error(f"Whisper load error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def __call__(self, audio):
        logger.debug("Whisper transcribing")
        try:
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
            logger.debug(f"Whisper transcription complete")
            return result['text']
        except Exception as e:
            logger.error(f"Whisper error: {type(e).__name__}: {e}", exc_info=True)
            raise
