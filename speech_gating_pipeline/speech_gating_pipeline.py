import re
import logging

from modules import Vad, Stt, Filter

logger = logging.getLogger(__name__)

class SpeechGatingPipeline:

    def __init__(self):
        logger.info("Initializing SpeechGatingPipeline")
        try:
            logger.debug("Initializing VAD (Voice Activity Detection)...")
            self.vad = Vad()
            self.should_stop = False
            logger.debug("✓ VAD initialized successfully")
            
            logger.debug("Loading Stt speech recognition model...")
            self.stt = Stt()
            logger.debug("✓ STT model loaded successfully")

            self.filter = Filter()
            
            logger.info("✓ SpeechGatingPipeline module initialized successfully")
        except Exception as e:
            logger.error(f"✗ SpeechGatingPipeline init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def listen(self) -> str:
        logger.info("🎤 SpeechGatingPipeline listening started")
        attempt = 0
        while True:
            try:
                attempt += 1
                logger.debug(f"Listen attempt #{attempt}")
                
                logger.debug("Waiting for audio capture...")
                audio = self.vad.listen()
                if audio is None:
                    continue
                logger.info(f"📥 Audio captured | size={len(audio)} samples | duration={len(audio)/16000:.2f}s")
                
                logger.debug("Processing audio with Whisper...")
                speach_text = self.stt(audio)

                filter_aproved = self._filter_stt(speach_text)

                if filter_aproved:
                    result = {
                        'text': speach_text,
                        "speaker": "Владелец",
                        'language': 'ru'
                    }
                    logger.info(f"✓ Speech recognized successfully | text_len={len(speach_text)}")
                    logger.debug(f"Recognized text: '{speach_text[:50]}...'")
                    return result
                    
            except TypeError as e:
                logger.warning(f"⚠ TypeError in listen loop (attempt #{attempt}): {e} - Continuing...")
                continue
            except Exception as e:
                logger.error(f"✗ Unexpected error in listen (attempt #{attempt}): {type(e).__name__}: {e}", exc_info=True)
                raise

    def _filter_stt(self, text: str) -> bool:
        return self.filter.filter_stt(text)



if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("SpeechGatingPipeline Module Standalone Test")
    logger.info("=" * 50)

    try:
        logger.info("Creating SpeechGatingPipeline instance...")
        pipeline = SpeechGatingPipeline()
        
        logger.info("Starting to listen...")
        result = pipeline.listen()
        
        logger.info(f"Test completed | Result: {result}")
        print(result)
    except Exception as e:
        logger.error(f"Test failed: {type(e).__name__}: {e}", exc_info=True)
    
    logger.info("=" * 50)