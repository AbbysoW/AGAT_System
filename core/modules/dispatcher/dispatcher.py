import logging
from threading import Thread
import asyncio

from client import send_rag, send_math, send_console
from .need_console import NeedConsole
from .need_math import NeedMath

logger = logging.getLogger(__name__)


class Dispatcher:

    def __init__(self):
        logger.info("Initializing Dispatcher module...")
        try:
            logger.debug("Loading NeedMath model...")
            self.need_math = NeedMath()
            logger.debug("✓ NeedMath model loaded")
            
            logger.debug("Loading NeedConsole model...")
            self.need_console = NeedConsole()
            logger.debug("✓ NeedConsole model loaded")
            
            logger.info("✓ Dispatcher initialized successfully")
        except Exception as e:
            logger.error(f"✗ Dispatcher init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def get_ext_inf(self, input: dict) -> dict: # Gets extra information from side modules
        logger.info("📤 Getting extra information from side modules (RAG, Math, Console)")
        logger.debug(f"Input text: '{input['stt']['data']['text']}'")
        
        threads: list[Thread] = []
        results: dict = {
            "rag": None,
            "math": None,
            "console": None
        }
        
        logger.debug("Starting parallel requests to side modules...")
        threads.append(Thread(target=self._send_rag, args=(input['stt']['data']['text'],results), name="RAG-Thread"))
        threads.append(Thread(target=self._send_math, args=(input['stt']['data']['text'],results), name="Math-Thread"))
        threads.append(Thread(target=self._send_console, args=(input['stt']['data']['text'],results), name="Console-Thread"))

        for thread in threads:
            thread.start()
            logger.debug(f"Thread started: {thread.name}")

        for thread in threads:
            thread.join()
            logger.debug(f"Thread completed: {thread.name}")

        logger.info(f"✓ Extra info results | rag={results['rag'] is not None} | math={results['math'] is not None} | console={results['console'] is not None}")
        logger.debug(f"Extra info details: {results}")
        return results

    # client functions
    def _send_rag(self, text: str, results: dict): # Sends request to rag system
        logger.debug("RAG request started")
        return
        try:
            result = asyncio.run(send_rag(text))
            results['rag'] = result # Will return RAG info
            logger.debug("✓ RAG request completed successfully")
        except Exception as e:
            logger.error(f"✗ RAG error: {type(e).__name__}: {e}", exc_info=True)

    def _send_math(self, text: str, results: dict): # Sends request to math system
        logger.debug("Checking if Math module is needed...")
        if self._need_math(text):
            logger.info("Math calculation needed - sending request")
            try:
                result = asyncio.run(send_math(text))
                results['math'] = result # Will retern an answer for math problem
                logger.debug("✓ Math request completed successfully")
            except Exception as e:
                logger.error(f"✗ Math error: {type(e).__name__}: {e}", exc_info=True)
        else:
            logger.debug("Math module not needed for this input")

    def _send_console(self, text: str, results: dict): # Sends request to console system
        logger.debug("Checking if Console module is needed...")
        if self._need_console(text):
            logger.info("Console command detected - sending request")
            try:
                result = asyncio.run(send_console(text)) # Will return true if can init comand else false
                results['console'] = result
                logger.debug("✓ Console request completed successfully")
            except Exception as e:
                logger.error(f"✗ Console error: {type(e).__name__}: {e}", exc_info=True)
        else:
            logger.debug("No console command detected")

    # filter functions
    def _need_math(self, text: str) -> bool: # Checks if math calculations are needed
        logger.debug("Running _need_math filter...")
        return
        try:
            y = self.need_math.filter(text)
            logger.debug(f"Math filter score: {y:.2f}")
            if y > 0.8:
                logger.info(f"✓ Math needed | confidence={y:.2f}")
                return True
            logger.debug(f"Math confidence too low | score={y:.2f} | threshold=0.8")
            return False
        except Exception as e:
            logger.error(f"✗ Math filter error: {type(e).__name__}: {e}", exc_info=True)
            return False

    def _need_console(self, text: str) -> bool: # Checks if console comands are needed
        logger.debug("Running _need_console filter...")
        return
        try:
            y = self.need_console.filter(text)
            logger.debug(f"Console filter score: {y:.2f}")
            if y > 0.8:
                logger.info(f"✓ Console needed | confidence={y:.2f}")
                return True
            logger.debug(f"Console confidence too low | score={y:.2f} | threshold=0.8")
            return False
        except Exception as e:
            logger.error(f"✗ Console filter error: {type(e).__name__}: {e}", exc_info=True)
            return False