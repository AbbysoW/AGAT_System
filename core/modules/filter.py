import logging
from .models.filter_models import KeepSilent, IsImportant

logger = logging.getLogger(__name__)


class Filter:
    class   Referee:
        keep_silent = KeepSilent()
        is_important = IsImportant()

    def __init__(self):
        logger.info("Filter initialized")
        self.reveree = self.Referee()

    def need_answer(self, input: dict) -> bool: # Checking if answer is needed
        logger.debug("Filtering input")
        return True

        if not self._should_keep_silent(input):
            if self._is_important(input):
                logger.debug("Input approved")
                return True

        logger.debug("Input rejected")
        return False
        
    def _should_keep_silent(self, input: dict) -> bool: # Checking if input should be ignored
        logger.debug("Checking keep_silent")
        try:
            y = self.reveree.keep_silent.filter(input['stt']['data']['text']) # gets prediction from the model

            if y > 0.8:
                logger.debug(f"Should keep silent | score={y:.2f}")
                return True
            return False
        except Exception as e:
            logger.error(f"Keep_silent error: {type(e).__name__}: {e}", exc_info=True)
            return False
    
    def _is_important(self, input: dict) -> bool: # Checking which input is important
        logger.debug("Checking importance")
        try:
            y = self.reveree.is_important.filter(input) # gets prediction from the model

            input['stt']['importance'] = y[0]
            input['cv']['importance'] = y[1]
            input['sys']['importance'] = y[2]
            
            logger.debug(f"Importance scores | stt={y[0]:.2f} cv={y[1]:.2f} sys={y[2]:.2f}")

            if max(y) > 0.8:
                logger.debug("Input is important")
                return True
            return False
        except Exception as e:
            logger.error(f"Importance check error: {type(e).__name__}: {e}", exc_info=True)
            return False