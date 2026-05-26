import logging
from .models import OwnModel

logger = logging.getLogger(__name__)

class SttImpModel:
    """Speech-to-Text Importance Model - determines if input is important."""
    
    def __init__(self):
        logger.info("Initializing SttImpModel...")
        try:
            logger.debug("Loading Model...")
            self.model = OwnModel()
            logger.info("✓ SttImpModel initialized successfully")
        except Exception as e:
            logger.error(f"✗ SttImpModel init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def __call__(self, input: dict, context: list[dict]) -> float:
        """Calculate importance score for input."""
        logger.debug(f"SttImpModel call | input_text='{input.get('text', 'N/A')[:50]}...' | context_size={len(context)}")
        
        try:
            logger.debug("Pre-processing input...")
            processed_input = self._preprocess(input, context)
            logger.debug(f"Pre-processing complete | processed_len={len(processed_input)}")
            
            logger.debug("Running model inference...")
            score = self.model(processed_input)
            logger.info(f"✓ Importance score calculated | score={score:.3f}")
            logger.debug(f"Processed input: {processed_input}")
            
            return score
        except Exception as e:
            logger.error(f"✗ Error in SttImpModel: {type(e).__name__}: {e}", exc_info=True)
            return 0.0

    def _preprocess(self, input: dict, context: list[dict]) -> str:
        """Preprocess input data into model format."""
        logger.debug("Pre-processing context...")
        processed_input = ""
        
        logger.debug(f"Processing context | size={len(context)}")
        for idx, el in enumerate(reversed(context)):
            logger.debug(f"Context element {idx+1}: /user='{el.get('/user', '')[:30]}...' | /model='{el.get('/model', '')[:30]}...'")
            processed_input += '/user ' + el.get("/user", "") + "; "
            processed_input += '/model ' + el.get("/model", "") + "; "

        logger.debug(f"Adding current input | text='{input.get('text', 'N/A')[:50]}...'")
        processed_input += '/user '
        processed_input += input.get('text', '') + ";"
        
        result = processed_input.strip()
        logger.debug(f"Pre-processing complete | output_len={len(result)}")
        return result


if __name__  == '__main__':
    logger.info("=" * 60)
    logger.info("SttImpModel Standalone Tests")
    logger.info("=" * 60)
    
    try:
        model = SttImpModel()
        
        # Control test
        logger.info("Test 1: Control test")
        context = { 
            '/user': 'Владелец: Включи свет в спальне',
            '/model': 'Свет в спальне включен'
        }

        input = {
            'speaker': 'Владелец',
            'text': 'Сделай яркость 20%'
        }

        # /user Владелец: Включи свет в спальне; /model Свет в спальне включен; /user Владелец: Сделай яркость 20%;
        result1 = model(input, [context])
        logger.info(f"Test 1 result: {result1}")
        print(result1) # ~1

        # test 1
        logger.info("Test 2: Related context test")
        context = { 
            '/user': 'Владелец: Агат, как твои дела?',
            '/model': 'У меня всё хорошо, спасибо! А как у тебя?'
        }

        input = {
            'speaker': 'Владелец',
            'text': 'Да всё хорошо'
        }

        result2 = model(input, [context])
        logger.info(f"Test 2 result: {result2}")
        print(result2) # ~1

        # test 2
        logger.info("Test 3: No context test")
        input = {
            'speaker': 'Владелец',
            'text': 'Агат, как твои дела?'
        }

        result3 = model(input, [])
        logger.info(f"Test 3 result: {result3}")
        print(result3) # ~1
        
        logger.info("All tests completed")
    except Exception as e:
        logger.error(f"Test failed: {type(e).__name__}: {e}", exc_info=True)
    
    logger.info("=" * 60)