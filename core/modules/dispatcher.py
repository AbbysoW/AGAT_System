from threading import Thread
import asyncio

from ..client import send_rag, send_math, send_console
from .models.dispatcher_models import NeedConsole, NeedMath


class Dispatcher:

    def __init__(self):
        self.need_math = NeedMath()
        self.need_console = NeedConsole()

    def get_ext_inf(self, input: dict) -> dict: # Gets extra information from side modules

        threads: list[Thread] = []
        results: dict = {
            "rag": "",
            "math": ""
        }
        
        threads.append(Thread(target=self._send_rag, args=(input['context'],results)))
        threads.append(Thread(target=self._send_math, args=(input['context'],results)))
        threads.append(Thread(target=self._send_console, args=(input['context'],)))

        for thread in threads:
            thread.start()
            thread.join()

        return results

    # client functions
    def _send_rag(self, text: str, results: dict): # Sends request to rag system
        return
        result = asyncio.run(send_rag(text))

        results['rag'] = result

    def _send_math(self, text: str, results: dict): # Sends request to math system
        if self._need_math(text):
            result = asyncio.run(send_math(text))

            results['math'] = result

    def _send_console(self, text: str): # Sends request to console system
        if self._need_console(text):
            asyncio.run(send_console(text))
    
    # filter functions
    def _need_math(self, text: str) -> bool: # Checks if math calculations are needed
        return
        y = self.need_math.filter(text)

        if y > 0.8:
            return True
        return False

    def _need_console(self, text: str) -> bool: # Checks if console comands are needed
        return
        y = self.need_math.filter(text)

        if y > 0.8:
            return True
        return False