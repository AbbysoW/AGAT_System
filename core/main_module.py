from modules.filter import Filter
from modules.dispatcher import Dispatcher
from modules.pre_process import PreProcess
from client import send_request






class Core:
    data = {
        "modules_online": {
            "stt": False,
            "cv": False,
            "sys": False
        },
        "input": {
            "context": "",
            "stt": "",
            "cv": {
                "text": "",
                "objects": []
            },
            "sys": {
                
            }

        }
    }

    def __init__(self):
        self.filter = Filter()
        self.dispatcher = Dispatcher()
        self.pre_process = PreProcess()

    def add_stt_input(self, text):

        self.data['input']['stt'] = text
        self.data['input']['context'] += text + " "

        self._check_inputs()

    


    def _check_inputs(self):
        if self.filter.need_answer(self.data['input']):
            ext_inf = self.dispatcher.get_ext_inf(self.data['input'])

            final_input = self.pre_process.process(self.data['input'], ext_inf)