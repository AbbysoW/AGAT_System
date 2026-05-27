import logging

from numpy import ndarray

from .volume_detector import VolumeDetector
from .speach_detector import SpeechDetector

logger = logging.getLogger(__name__)

class Vad:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        self.volume_detector = VolumeDetector(sample_rate)
        self.speech_detector = SpeechDetector(sample_rate)

    def listen(self) -> ndarray:
        logger.info("🎤 VAD listening started")
        audio = self.volume_detector()
        is_speech = self.speech_detector(audio)
        if is_speech:
            logger.debug("Speech detected")
            return audio
        else:
            logger.debug(f"⚠ Low confidence: is_speech prob < 0.6, -> {is_speech}, retrying...")


 # получит аудио от volume_detector
 # speach_detector проверит айдио
 # Вернет это это же аудио 

#  logger.info("🎤 VAD listening started")