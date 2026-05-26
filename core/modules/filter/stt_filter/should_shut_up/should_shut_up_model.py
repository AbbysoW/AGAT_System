from .models import PrototypeBasedClassifier

import logging


logger = logging.getLogger(__name__)

class ShouldShutUpModel:
    
    def __init__(self):
        logger.info("Initializing ShouldShutUpModel...")
        try:
            logger.debug("Loading Model...")
            self.model = PrototypeBasedClassifier()
            logger.info("✓ ShouldShutUpModel initialized successfully")
        except Exception as e:
            logger.error(f"✗ ShouldShutUpModel init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def __call__(self, input: dict) -> float:
        logger.debug(f"ShouldShutUpModel call | input_text='{input.get('text', 'N/A')[:50]}...'")

        try:
            logger.debug("Pre-processing input...")
            processed_input = self._preprocess(input)
            logger.debug(f"Pre-processing complete | processed_len={len(processed_input)}")
            
            logger.debug("Running model inference...")
            score = self.model(processed_input)
            logger.info(f"✓ Importance score calculated | score={score:.3f}")
            logger.debug(f"Processed input: {processed_input}")
            
            return score
        except Exception as e:
            logger.error(f"✗ Error in ShouldShutUpModel: {type(e).__name__}: {e}", exc_info=True)
            return 0.0
        
    def _preprocess(self, input: dict) -> str:
        """Preprocess input data into model format."""
        print(input)
        logger.debug("Pre-processing input...")
        processed_input = input.get("text", "")
        logger.debug(f"Pre-processing complete | output_len={len(processed_input)}")
        return processed_input
