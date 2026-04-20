from datetime import datetime

class PreProcess:
    
    def __init__(self):
        pass

    def process(self, input: dict, extra: dict) -> str: # Deletes old, not important data and noisy data
        #input = self._noise_filter(input) Bring back  when filter is done

        input['sys'] = self._data_ennrichment(input['sys'])

        concatenated_input = self._concat_inputs(input, extra)

        return concatenated_input
    
    
    def _noise_filter(self, input: dict) -> dict:

        # delete unimportant data
        filtered_input = self._delete_unimportant_data(input)

        # delete old data
        filtered_input = self._delete_old_data(filtered_input)

        return filtered_input
    
    def _delete_unimportant_data(self, input: dict) -> dict:
        filtered_input = {}
        for key in input:
            if input[key]['importance'] < 0.5:
                filtered_input[key] = input[key]
        return filtered_input
    
    def _delete_old_data(self, input: dict) -> dict: # Deletes data depending on the type of data
        filtered_input = {}
        current_time = datetime.now().timestamp()

        # is stt old
        if 'stt' in input and current_time - input['stt']['last_update'] < 3600: # 1 hour
            filtered_input['stt'] = input['stt']

        # is cv old
        if 'cv' in input and current_time - input['cv']['last_update'] < 600: # 10 minutes
            filtered_input['cv'] = input['cv']

        # is sys old
        if 'sys' in input and current_time - input['sys']['last_update'] < 86400: # 1 day
            filtered_input['sys'] = input['sys']

        return filtered_input

    def _data_ennrichment(self, system_input: dict) -> dict:
        # Placeholder for data enrichment logic
        return system_input
    
    def _concat_inputs(self, input: dict,extra: dict) -> str:
        pass


