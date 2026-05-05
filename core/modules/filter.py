from .models.filter_models import KeepSilent, IsImportant


class Filter:
    class   Referee:
        keep_silent = KeepSilent()
        is_important = IsImportant()

    def __init__(self):
        self.reveree = self.Referee()

    def need_answer(self, input: dict) -> bool: # Checking if answer is needed
        return True

        if not self._should_keep_silent(input):
            if self._is_important(input):
                return True

        return False
        
    def _should_keep_silent(self, input: dict) -> bool: # Checking if input should be ignored
        y = self.reveree.keep_silent.filter(input['stt']['data']['text']) # gets prediction from the model

        if y > 0.8:
            return True
        return False
    
    def _is_important(self, input: dict) -> bool: # Checking which input is important
        y = self.reveree.is_important.filter(input) # gets prediction from the model

        input['stt']['importance'] = y[0]
        input['cv']['importance'] = y[1]
        input['sys']['importance'] = y[2]

        if max(y) > 0.8:
            return True
        return False