from datetime import datetime
from typing_extensions import final

class PreProcess:
    
    def __init__(self):
        pass

    def process(self, input: dict, extra: dict) -> tuple[dict, dict]:
        # input: data['input']
        
        new_context_el = self._make_context_element(input)
        processed_extra = self._process_extra(extra)

        model_input = self._add_extra(new_context_el, processed_extra)

        return model_input, new_context_el

    def _make_context_element(self, input: dict) -> dict:
        final_context = {
            '/sys': self._sys_process_data(input['sys']),
            '/cv': self._cv_process_data(input['cv']),
            '/user': self._stt_process_data(input['stt']),
        }
        return final_context

    def _stt_process_data(self, stt_data: dict):
        if stt_data['importance'] > 0.8:
            return f"{stt_data['data']['speaker']}: {stt_data['data']['text']}"
        return None
    
    def _cv_process_data(self, cv_data: dict):
        if cv_data['importance'] > 0.8:
            return ""
        return None
    
    def _sys_process_data(self, sys_data: dict):
        if sys_data['importance'] > 0.8:    
            return ""
        return None
        
    def _add_extra(self, data: dict, extra: str):
        new_context = data.copy()
        new_context.update({'/hint': extra})

        return new_context
    
    def _process_extra(self, extra: dict) -> str:
        # extra: dict = { # reference
        #     "rag": None,
        #     "math": None,
        #     "console": None
        # }

        str_extra = ''

        for k, v in extra.items():
            if v is not None:
                str_extra += f"{k}: {v}\n"
        return str_extra
    
# Хз будто чего-то не хватает
# Ну будто минимум сделан чтоб эта тарантайка завелась