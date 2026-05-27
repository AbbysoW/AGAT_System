import logging

from numpy import ndarray

from .models import SileroVad


logger = logging.getLogger(__name__)


class SpeechDetector:
    
    def __init__(self, sample_rate: int = 16000):
        self.model = SileroVad(sample_rate)

    def __call__(self, audio: ndarray) -> bool:
        logger.info("Calculating speech probability...")
        prob = self.model(audio)
        logger.info(f"Speech prob calculated: {prob:.2f}")
        return prob > 0.6