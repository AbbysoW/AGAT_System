import re
import logging

from pipeline.models import Whisper
from pipeline.vad import VAD

logger = logging.getLogger(__name__)

class STT:

    _PATTERN = re.compile(r'[A-Za-z0-9\u0400-\u04FF]')
    
    def __init__(self, samplerate: int = 16000):
        logger.info(f"STT init | samplerate={samplerate}")
        try:
            self.vad = VAD(samplerate)
            self.should_stop = False
            logger.debug("VAD initialized")
            
            self.medium = Whisper("medium")
            logger.debug("Whisper model loaded")
        except Exception as e:
            logger.error(f"STT init error: {type(e).__name__}: {e}", exc_info=True)
            raise
    
    def _get_speach_text(self, audio):
        logger.debug("Whisper processing started")
        try:
            result = self.medium(audio)
            logger.debug(f"Whisper result | lang={result.get('language')} | text_len={len(result.get('text', ''))}")
            return result
        except Exception as e:
            logger.error(f"Whisper error: {type(e).__name__}: {e}", exc_info=True)
            raise
    
    def _clacuate_speech_probability(self, segments):
        sum = 0
        for seg in segments:
            sum += seg['no_speech_prob']
        return 1 - (sum / len(segments))

    def listen(self) -> str:
        logger.info("STT listen started")
        attempt = 0
        while True:
            try:
                attempt += 1
                audio = self.vad.listen()
                logger.debug(f"Audio captured | size={len(audio)}")
                
                speach_info = self._get_speach_text(audio)

                if speach_info['text'] and self._PATTERN.search(speach_info['text']):
                    confidence = self._clacuate_speech_probability(speach_info['segments'])
                    logger.debug(f"Speech confidence={confidence:.2f}")
                    
                    if confidence > 0.6:
                        result = {
                            'text': speach_info['text'],
                            "speaker": "Владелец",
                            'language': speach_info['language']
                        }
                        logger.info(f"Speech recognized | lang={result['language']} | text_len={len(result['text'])}")
                        return result
                    else:
                        logger.debug(f"Low confidence, retrying")
                else:
                    logger.debug("No speech detected")
                    
            except TypeError as e:
                logger.warning(f"TypeError in listen loop (attempt {attempt}): {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in listen (attempt {attempt}): {type(e).__name__}: {e}", exc_info=True)
                raise
                

if __name__ == "__main__":

    pipeline = STT()

    print(pipeline.listen())