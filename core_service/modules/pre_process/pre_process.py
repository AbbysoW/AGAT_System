import logging


logger = logging.getLogger(__name__)

class PreProcess:
    
    def __init__(self):
        logger.info("PreProcess module initialized")
        logger.debug("PreProcess is ready to process input data")

    def process(self, input: dict, extra: dict) -> tuple[dict, dict]:
        logger.info("🔄 Pre-processing input data")
        logger.debug(f"Input keys: {list(input.keys())} | Extra keys: {list(extra.keys())}")
        
        try:
            logger.debug("Creating context element...")
            new_context_el = self._make_context_element(input)
            logger.debug(f"Context element created | keys: {list(new_context_el.keys())}")

            logger.debug("Processing extra information...")
            processed_extra = self._process_extra(extra)
            logger.debug(f"Extra information processed | content_len={len(processed_extra)}")

            logger.debug("Adding extra information to context...")
            model_input = self._add_extra(new_context_el, processed_extra)
            logger.debug(f"Model input prepared | keys: {list(model_input.keys())}")

            logger.info(f"✓ Pre-processing completed successfully | model_input: {model_input}")

            return model_input, new_context_el
        except Exception as e:
            logger.error(f"✗ Pre-process error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _make_context_element(self, input: dict) -> dict:
        logger.debug("Creating context element from input")
        
        sys_data = self._sys_process_data(input['sys'])
        logger.debug(f"System data processed | important={input['sys']['importance']} | content='{sys_data}'")
        
        cv_data = self._cv_process_data(input['cv'])
        logger.debug(f"CV data processed | important={input['cv']['importance']} | content='{cv_data}'")
        
        stt_data = self._stt_process_data(input['stt'])
        logger.debug(f"STT data processed | important={input['stt']['importance']} | content='{stt_data}'")
        
        final_context = {
            '/sys': sys_data,
            '/cv': cv_data,
            '/user': stt_data,
        }
        
        logger.debug(f"Final context element: {final_context}")
        return final_context

    def _stt_process_data(self, stt_data: dict):
        logger.debug(f"Processing STT data | importance={stt_data['importance']}")
        if stt_data['importance']:
            result = f"{stt_data['data']['speaker']}: {stt_data['data']['text']}"
            logger.debug(f"STT data marked important | result: {result}")
            return result
        logger.debug("STT data marked unimportant | returning None")
        return None
    
    def _cv_process_data(self, cv_data: dict):
        logger.debug(f"Processing CV data | importance={cv_data['importance']}")
        if cv_data['importance']:
            logger.debug("CV data marked important | returning empty string")
            return ""
        logger.debug("CV data marked unimportant | returning None")
        return None
    
    def _sys_process_data(self, sys_data: dict):
        logger.debug(f"Processing System data | importance={sys_data['importance']}")
        if sys_data['importance']:    
            logger.debug("System data marked important | returning empty string")
            return ""
        logger.debug("System data marked unimportant | returning None")
        return None
        
    def _add_extra(self, data: dict, extra: str):
        logger.debug(f"Adding extra information to context | extra_len={len(extra)}")
        new_context = data.copy()
        new_context.update({'/hint': extra})
        logger.debug(f"Extra information added | new_keys: {list(new_context.keys())}")

        return new_context
    
    def _process_extra(self, extra: dict) -> str:
        """Process extra information from dispatcher (RAG, Math, Console)."""
        logger.debug(f"Processing extra information | keys: {list(extra.keys())}")

        str_extra = ''

        for k, v in extra.items():
            if v is not None:
                logger.debug(f"Adding {k} to extra: {v}")
                str_extra += f"{k}: {v}\n"
            else:
                logger.debug(f"Skipping {k} (None)")
        
        logger.debug(f"Extra information processed | final_len={len(str_extra)} | content: {str_extra[:100]}...")
        return str_extra