from datetime import datetime

from whisper_stt import Whisper
from vad import VAD

class Pipeline:
    vad = VAD()
    small = Whisper("small")

    def __init__(self):
        pass
    
    def _get_speach_text(self, audio):
        medium = Whisper("medium")
        return medium.get_text(audio)
    
    def listen(self):

        while True:
            try:
                audio = self.vad.listen()
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
