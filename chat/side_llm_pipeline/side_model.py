import logging
from .models import Qwen3_4B

logger = logging.getLogger(__name__)

class SideModel:
    hierarchy =  [
        '/hint',
        '/sys',
        '/cv',
        '/user',
        '/model'
    ]

    def __init__(self):
        logger.info("SideModel init")
        try:
            self.pipeline = Qwen3_4B()
            logger.info("Qwen3 model loaded")
        except Exception as e:
            logger.error(f"SideModel init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _preprocess(self, context_list: list[dict]) -> str:
        # context_list[-1] = {
        #     '/hint': '',
        #     '/sys': '',
        #     '/cv': '',
        #     '/user': '',
        #     '/model': ''
        # }

        final_context = ""
        for con_el in context_list:
            final_context += self._distribute_roller(con_el)

        return final_context
    
    def _distribute_roller(self, con_el: dict) -> str:
        promt = ""

        for role in self.hierarchy:
            promt += self._get_role_info(role, con_el)

        return promt
            
    def _get_role_info(self, role: str, con_el: dict) -> str:
        standart_message = "(empty)"

        if con_el.get(role):
            return role + '\n' + con_el[role] + '\n'
        else:
            return role + '\n' + standart_message + '\n'

    def __call__(self, x: list[dict]) -> str:
        logger.info(f"SideModel call | context_len={len(x)}")
        try:
            promt = self._preprocess(x)
            logger.debug(f"Prompt created | length={len(promt)}")
            result = self.pipeline(promt)
            logger.info(f"Pipeline result | length={len(str(result))}")
            return result
        except Exception as e:
            logger.error(f"SideModel error: {type(e).__name__}: {e}", exc_info=True)
            raise