import logging
from side_llm_pipeline import SideModel

logger = logging.getLogger(__name__)

class ChatModule:
    def __init__(self):
        logger.info("Initializing ChatModule...")
        try:
            logger.debug("Loading SideModel...")
            self.model = SideModel()
            logger.info("✓ Chat model loaded successfully")
            logger.debug("ChatModule ready for inference")
        except Exception as e:
            logger.error(f"✗ ChatModule init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def __call__(self, x: list[dict]) -> str:
        logger.info(f"📤 Chat model call | context_len={len(x)}")
        logger.debug(f"Context structure: {[list(el.keys()) for el in x]}")
        
        try:
            logger.debug("Starting model inference...")
            result = self.model(x)
            
            logger.info(f"✓ Chat result | result_len={len(str(result))}")
            logger.debug(f"Chat result preview: {str(result)[:150]}..." if len(str(result)) > 150 else f"Chat result: {result}")
            
            return result
        except Exception as e:
            logger.error(f"✗ Chat error: {type(e).__name__}: {e}", exc_info=True)
            raise    
    
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("ChatModule Standalone Test")
    logger.info("=" * 60)

    context = [
        {
            '/hint': None,
            '/sys': None,
            '/cv': None,
            '/user': "Владелец: Привет Агат! Тестовый запуск. Посчитай от 1 до 5",
            '/model': None
        }
    ]

    logger.debug(f"Test context prepared: {context}")
    
    try:
        logger.info("Creating ChatModule instance...")
        chat = ChatModule()
        
        logger.info("Running inference...")
        result = chat(context)
        
        logger.info(f"Test completed | Result: {result}")
        print(result)
    except Exception as e:
        logger.error(f"Test failed: {type(e).__name__}: {e}", exc_info=True)
    
    logger.info("=" * 60)