import logging
from side_llm_pipeline import SideModel

logger = logging.getLogger(__name__)

class ChatModule:
    def __init__(self):
        logger.info("ChatModule init")
        try:
            self.model = SideModel()
            logger.info("Chat model loaded")
        except Exception as e:
            logger.error(f"ChatModule init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def __call__(self, x: list[dict]) -> str:
        logger.info(f"Chat call | context_len={len(x)}")
        try:
            result = self.model(x)
            logger.info(f"Chat result | result_len={len(str(result))}")
            return result
        except Exception as e:
            logger.error(f"Chat error: {type(e).__name__}: {e}", exc_info=True)
            raise    
    
if __name__ == '__main__':

    context = [
        {
            '/hint': None,
            '/sys': None,
            '/cv': None,
            '/user': "Владелец: Привет Агат! Тестовый запуск. Посчитай от 1 до 5",
            '/model': None
        }
    ]

    chat = ChatModule()
    print(chat(context))