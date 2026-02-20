from datetime import datetime

from whisper_stt import Whisper
from vad import VAD

class Pipeline:
    vad = VAD()
    small = Whisper("small")

    def __init__(self):
        pass

    def _is_AGAT_called(self, audio):
        text = self.small.get_text(audio)['text']

        AGAT_name = ['agat', 'агат']

        # print("text:", text)

        if any(name in text.lower() for name in AGAT_name) or True:#  if AGAT name is in text:
            return True
        
        print("AGAT is not called")
        return False
    
    def _get_speach_text(audio):
        medium = Whisper("medium")
        return medium.get_text(audio)
    
    def listen(self, dialogue_continues: bool = False):
        if dialogue_continues:
            start_waiting = datetime.now()

        while True:
            try:
                audio = self.vad.listen()

                if dialogue_continues:
                    delta_time = datetime.now() - start_waiting 
                    if delta_time.total_seconds() > 300:
                        dialogue_continues = False

                if dialogue_continues or self._is_AGAT_called(audio):
                    speach_text = self._get_speach_text(audio)
                    print("text:", speach_text)

                    if speach_text.text != "":
                        return speach_text
                    
                    print("nothing found")
            except TypeError:
                continue
                

if __name__ == "__main__":

    pipeline = Pipeline()

    print(pipeline.listen())
