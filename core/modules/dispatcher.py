import logging
from threading import Thread
import asyncio

from ..client import send_rag, send_math, send_console
from .models.dispatcher_models import NeedConsole, NeedMath

logger = logging.getLogger(__name__)


class Dispatcher:

    def __init__(self):
        logger.info("Dispatcher initialized")
        self.need_math = NeedMath()
        self.need_console = NeedConsole()

    def get_ext_inf(self, input: dict) -> dict: # Gets extra information from side modules
        logger.debug("Getting extra information")
        threads: list[Thread] = []
        results: dict = {
            "rag": None,
            "math": None,
            "console": None
        }
        
        threads.append(Thread(target=self._send_rag, args=(input['stt']['data']['text'],results)))
        threads.append(Thread(target=self._send_math, args=(input['stt']['data']['text'],results)))
        threads.append(Thread(target=self._send_console, args=(input['stt']['data']['text'],results)))

        for thread in threads:
            thread.start()
            thread.join()

        logger.debug(f"Extra info results | rag={results['rag'] is not None} math={results['math'] is not None} console={results['console'] is not None}")
        return results

    # client functions
    def _send_rag(self, text: str, results: dict): # Sends request to rag system
        logger.debug("RAG request started")
        return
        try:
            result = asyncio.run(send_rag(text))
            results['rag'] = result # Will return RAG info
            logger.debug("RAG request completed")
        except Exception as e:
            logger.error(f"RAG error: {type(e).__name__}: {e}")

    def _send_math(self, text: str, results: dict): # Sends request to math system
        logger.debug("Math check")
        if self._need_math(text):
            logger.debug("Math request started")
            try:
                result = asyncio.run(send_math(text))
                results['math'] = result # Will retern an answer for math problem
                logger.debug("Math request completed")
            except Exception as e:
                logger.error(f"Math error: {type(e).__name__}: {e}")

    def _send_console(self, text: str, results: dict): # Sends request to console system
        logger.debug("Console check")
        if self._need_console(text):
            logger.debug("Console request started")
            try:
                result = asyncio.run(send_console(text)) # Will return true if can init comand else false
                results['console'] = result
                logger.debug("Console request completed")
            except Exception as e:
                logger.error(f"Console error: {type(e).__name__}: {e}")

    # filter functions
    def _need_math(self, text: str) -> bool: # Checks if math calculations are needed
        logger.debug("Checking need_math")
        return
        try:
            y = self.need_math.filter(text)
            if y > 0.8:
                logger.debug(f"Math needed | score={y:.2f}")
                return True
            return False
        except Exception as e:
            logger.error(f"Math filter error: {type(e).__name__}: {e}")
            return False

    def _need_console(self, text: str) -> bool: # Checks if console comands are needed
        logger.debug("Checking need_console")
        return
        try:
            y = self.need_console.filter(text)
            if y > 0.8:
                logger.debug(f"Console needed | score={y:.2f}")
                return True
            return False
        except Exception as e:
            logger.error(f"Console filter error: {type(e).__name__}: {e}")
            return False