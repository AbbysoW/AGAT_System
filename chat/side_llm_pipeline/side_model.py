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
        logger.info("Initializing SideModel...")
        try:
            logger.debug(f"Loading Qwen3 model | hierarchy: {self.hierarchy}")
            self.pipeline = Qwen3_4B()
            logger.info("✓ Qwen3 model loaded successfully")
            logger.debug("SideModel ready for processing")
        except Exception as e:
            logger.error(f"✗ SideModel init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _preprocess(self, context_list: list[dict]) -> str:
        """Preprocess context list into a prompt string."""
        logger.debug(f"Pre-processing context | context_size={len(context_list)}")
        
        # context_list[-1] = {
        #     '/hint': '',
        #     '/sys': '',
        #     '/cv': '',
        #     '/user': '',
        #     '/model': ''
        # }

        final_context = ""
        for idx, con_el in enumerate(context_list):
            logger.debug(f"Processing context element {idx+1}/{len(context_list)}")
            element_str = self._distribute_roller(con_el)
            final_context += element_str
            logger.debug(f"Context element {idx+1} added | current_len={len(final_context)}")

        logger.debug(f"Context pre-processing completed | total_len={len(final_context)}")
        return final_context
    
    def _distribute_roller(self, con_el: dict) -> str:
        """Distribute roles and format context element."""
        logger.debug(f"Distributing roles for context element | roles: {list(con_el.keys())}")
        promt = ""

        for role in self.hierarchy:
            logger.debug(f"Processing role: {role}")
            role_info = self._get_role_info(role, con_el)
            promt += role_info
            logger.debug(f"Role {role} added | content_len={len(role_info)}")

        return promt
            
    def _get_role_info(self, role: str, con_el: dict) -> str:
        """Get information for a specific role."""
        standart_message = "(empty)"

        if con_el.get(role):
            content = con_el[role]
            logger.debug(f"Role {role} has content | len={len(str(content))}")
            return role + '\n' + content + '\n'
        else:
            logger.debug(f"Role {role} is empty - using default message")
            return role + '\n' + standart_message + '\n'

    def __call__(self, x: list[dict]) -> str:
        logger.info(f"📤 SideModel call | context_len={len(x)}")
        logger.debug(f"Context structure: {[list(el.keys()) for el in x]}")
        
        try:
            logger.debug("Starting prompt creation...")
            promt = self._preprocess(x)
            logger.info(f"✓ Prompt created | length={len(promt)}")
            logger.debug(f"Prompt preview:\n{promt[-200:]}..." if len(promt) > 200 else f"Prompt:\n{promt}")
            
            logger.debug("Starting pipeline inference...")
            result = self.pipeline(promt)
            logger.info(f"✓ Pipeline result received | length={len(str(result))}")
            logger.debug(f"Result preview: {str(result)[:150]}..." if len(str(result)) > 150 else f"Result: {result}")
            
            return result
        except Exception as e:
            logger.error(f"✗ SideModel error: {type(e).__name__}: {e}", exc_info=True)
            raise