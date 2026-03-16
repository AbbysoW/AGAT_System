import json

from models.filter_models import KeepSilent, IsImportant


class Filter:
    class Referee:
        keep_silent = KeepSilent()
        is_important = IsImportant()

    def __init__(self):
        self.reveree = self.Referee()

    def need_answer(self, input: dict) -> bool:
        inputs_str = input['stt'] # Временое решение пока не придумаю как предствить все инпуты

        if not self._should_keep_silent(inputs_str):
            if self._is_important(inputs_str):
                return True

        return False
        
    def _should_keep_silent(self, input: str) -> bool:
        y = self.reveree.keep_silent.filter(input)

        if y > 0.8:
            return True
        return False
    
    def _is_important(self, input: str) -> bool:
        y = self.reveree.is_important.filter(input)

        if max(y) > 0.8:
            return True
        return False