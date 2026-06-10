import logging

from .models import Whisper


logger = logging.getLogger(__name__)

class Stt:

    def __init__(self):
        self.model = Whisper("medium")

    def __call__(self, audio) -> str:
        logger.debug(f"STT processing started | audio_size={len(audio)} samples")
        result = self.model(audio)
        logger.debug(f"STT processing completed | text='{result[:50]}'")
        return result
