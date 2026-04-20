from datetime import datetime

from .modules.filter import Filter
from .modules.dispatcher import Dispatcher
from .modules.pre_process import PreProcess


class Core:
    data = {
        'modules_online': {
            'stt': False,
            'cv': False,
            'sys': False
        },
        'input': {
            'context': {
                'importance': 0,
                'data':{
                    'text': ""
                },
            },
            'stt': {
                'last_update': 0,
                'importance': 0,
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

            # final_input = self.pre_process.process(self.data['input'], ext_inf) # processing the final input

            print(self.data['input'])




if __name__ == '__main__':

    core = Core()

    core.add_stt_input("test stt input")