from .models import Qwen3_4B


class SideModel:
    hierarchy =  [
        '/hint',
        '/sys',
        '/cv',
        '/user',
        '/model'
    ]

    def __init__(self):
        self.pipeline = Qwen3_4B()

    def _preprocess(self, context_list: list[dict]) -> str:
        # context_list[-1] = {
        #     '/hint': '',
        #     '/sys': '',
        #     '/cv': '',
        #     '/user': '',
        #     '/model': ''
        # }

        final_context = ""
        for con_el in context_list:
            final_context += self._distribute_roller(con_el)

        return final_context
    
    def _distribute_roller(self, con_el: dict) -> str:
        promt = ""

        for role in self.hierarchy:
            promt += self._get_role_info(role, con_el)

        return promt
            
    def _get_role_info(self, role: str, con_el: dict) -> str:
        standart_message = "(empty)"

        if con_el.get(role):
            return role + '\n' + con_el[role] + '\n'
        else:
            return role + '\n' + standart_message + '\n'

    def __call__(self, x: list[dict]) -> str:
        promt = self._preprocess(x)
        return self.pipeline(promt)