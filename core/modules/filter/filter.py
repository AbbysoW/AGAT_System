import logging
from  threading import Thread

from .stt_filter import SttFilter

logger = logging.getLogger(__name__)


class Filter:

    dialogue_continues = False

    def __init__(self):
        logger.info("Initializing Filter module...")
        try:
            logger.debug("Loading STT filter model...")
            self.stt_filter = SttFilter()
            logger.debug("✓ STT filter loaded")
            
            self.cv_filter = None
            self.sys_filter = None
            logger.debug("CV filter: Not yet implemented")
            logger.debug("System filter: Not yet implemented")
            
            logger.info("✓ Filter module initialized successfully")
            logger.debug(f"Dialogue continuation mode: {self.dialogue_continues}")
        except Exception as e:
            logger.error(f"✗ Filter init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    # --- Input Filter ---
    # -- Input Importance -- 
    def get_input_importance(self, input: dict, context: list[dict]) -> dict[str, bool]:
        """Returns importans for all inputs"""
        logger.debug("=== Starting parallel filtering ===")
        threads: list[Thread] = []
        importance: dict[str, bool] = {}

        logger.debug("Creating STT filter thread...")
        threads.append(Thread(
            target=self._filter_stt,
            args=(input['stt'], context[-1:], importance),
            name="STT-Filter-Thread"
        ))
        logger.debug("Creating CV filter thread...")
        threads.append(Thread(
            target=self._filter_cv,
            args=(input['cv'], context, importance),
            name="CV-Filter-Thread"
        ))
        logger.debug("Creating System filter thread...")
        threads.append(Thread(
            target=self._filter_sys,
            args=(input['sys'], context, importance),
            name="Sys-Filter-Thread"
        ))

        logger.debug("Starting all filter threads...")
        for thread in threads:
            thread.start()
            logger.debug(f"Thread started: {thread.name}")

        logger.debug("Waiting for all filter threads to complete...")
        for thread in threads:
            thread.join()
            logger.debug(f"Thread completed: {thread.name}")

        logger.debug(f"=== Filtering complete | Results: {importance} ===")
        return importance
    
    def _filter_stt(self, input: dict, context: list[dict], importance: dict):
        logger.debug("STT filter: Analyzing speech input...")
        try:
            is_important: bool = self.stt_filter(input, context)
            logger.debug(f"STT filter result: {is_important}")
            importance.update({'stt': is_important})
        except Exception as e:
            logger.error(f"✗ STT filter error: {type(e).__name__}: {e}", exc_info=True)
            importance.update({'stt': False})

    def _filter_cv(self, input: dict, context: list[dict], importance: dict):
        logger.debug("CV filter: Not yet implemented - marking as unimportant")
        importance.update({'cv': False})

    def _filter_sys(self, input: dict, context: list[dict], importance: dict):
        logger.debug("System filter: Not yet implemented - marking as unimportant")
        importance.update({'sys': False})

    # -- Should Shut Up -- 
    def should_shut_up(self, input: dict) -> bool:
        should_shut = self.stt_filter.should_shut_up(input['stt']['data'])
        return should_shut


    # --- Dispather Filter ---