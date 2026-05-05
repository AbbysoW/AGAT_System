import asyncio
from datetime import datetime

from .modules.filter import Filter
from .modules.dispatcher import Dispatcher
from .modules.pre_process import PreProcess
from .client import send_request


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
        self.filter = Filter()
        self.dispatcher = Dispatcher()
        self.pre_process = PreProcess()

    def add_stt_input(self,text, speaker, language): # Updating STT input data
        self.data['input']['stt']['last_update'] = datetime.now().timestamp() # undoating time 
        self.data['input']['stt']['data'] = { # updating data
            'text': text,
            'language': language,
            'speaker': speaker
        }
        self._check_inputs()


    def _check_inputs(self): # Checking new input importance 
        if self.filter.need_answer(self.data['input']):
            # Bring back when modulest will be done
            # ext_inf = self.dispatcher.get_ext_inf(self.data['input']) # geting extra information for answer

            final_input, final_context = self.pre_process.process(self.data['input'], {}) # processing the final input

            # print(self.data['input'])

            model_answer = asyncio.run(send_request(self.data['context'] + [final_input]))
            if model_answer:
                self._update_context(final_context, model_answer)


            #  Дальше должен быть анализ диалога для пополнения RAG

    def _update_context(self, last_context: dict, model_answer: str):
        last_context.update({'/model': model_answer})
        self.data['context'].append(last_context)




if __name__ == '__main__':

    core = Core()

    core.add_stt_input("test stt input")