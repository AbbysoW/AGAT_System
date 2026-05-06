import asyncio
import logging
from datetime import datetime

from modules.filter import Filter
from modules.dispatcher import Dispatcher
from modules.pre_process import PreProcess
from client import send_request

logger = logging.getLogger(__name__)

class Core:
    data = {
        'modules_online': {
            'stt': False,
            'cv': False,
            'sys': False
        },
        # ПОТРЕБУЕТ ЧИСТКИ ПО ДОСТИЖЕНИЮ ОПРЕДЕЛЕННОЙ ДЛИНЫ. 
        # Надо будет написать суммаризатор для этого
        # В дальнейшей обработке я использую только последный контектс
        # Могу спокой делать краткую выжимку из половины ддмиалога и не сломаю систему даже))))))))
        'context': [ 
            # { # Just reference
            #     '/sys': '',
            #     '/cv': '',
            #     '/user': '',
            #     '/model': ''
            # }
        ],
        'input': {
            'stt': {
                'last_update': 0,
                'importance': 1,
                'data':{
                    "speaker": "Владелец",
                    "language": "ru",
                    'text': ""
                }
            },
            'cv': {
                'last_update': 0,
                'importance': 0,
                'data':{
                    'text': "",
                    'objects': []
                }
            },
            'sys': {
                'last_update': 0,
                'importance': 0,
                'data': {

                }
            }
        }
    }

    def __init__(self):
        logger.info("Core init")
        try:
            self.filter = Filter()
            self.dispatcher = Dispatcher()
            self.pre_process = PreProcess()
            logger.info("Core modules initialized")
        except Exception as e:
            logger.error(f"Core init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def add_stt_input(self,text, speaker, language): # Updating STT input data
        logger.info(f"STT input | speaker={speaker} | lang={language} | text_len={len(text)}")
        self.data['input']['stt']['last_update'] = datetime.now().timestamp() # undoating time 
        self.data['input']['stt']['data'] = { # updating data
            'text': text,
            'language': language,
            'speaker': speaker
        }
        self._check_inputs()


    def _check_inputs(self): # Checking new input importance 
        logger.debug("Checking input importance")
        if self.filter.need_answer(self.data['input']):
            logger.info("Input approved by filter")
            # Bring back when modulest will be done
            # ext_inf = self.dispatcher.get_ext_inf(self.data['input']) # geting extra information for answer

            try:
                final_input, final_context = self.pre_process.process(self.data['input'], {}) # processing the final input
                logger.debug("Pre-processing completed")

                logger.info("Sending request to chat service")
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                model_answer = loop.run_until_complete(
                    send_request(self.data['context'] + [final_input])
                )
                
                if model_answer:
                    logger.info(f"Chat response received | answer_len={len(str(model_answer))}")
                    self._update_context(final_context, model_answer)
                    logger.debug("Context updated")
                else:
                    logger.warning("Chat service returned None")
                    
            except Exception as e:
                logger.error(f"Chat request error: {type(e).__name__}: {e}", exc_info=True)
            #  Дальше должен быть анализ диалога для пополнения RAG
        else:
            logger.debug("Input rejected by filter")

    def _update_context(self, last_context: dict, model_answer: str):
        last_context.update({'/model': model_answer})
        self.data['context'].append(last_context)




if __name__ == '__main__':

    core = Core()

    core.add_stt_input("test stt input")