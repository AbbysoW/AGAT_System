
from .whisper_stt import Whisper
from .vad import VAD

class STT:
    
    def __init__(self):
        self.vad = VAD()

        # self.small = Whisper("small")
        self.medium = Whisper("small")
    
    def _get_speach_text(self, audio):
        return self.medium.get_text(audio)
    
    def listen(self):

        while True:
            try:
                audio = self.vad.listen()
                speach_info = self._get_speach_text(audio)
                print("text:", speach_info)

                if speach_info['text']:
                    return speach_info['text']
                
                print("nothing found")
            except TypeError:
                continue
                

if __name__ == "__main__":

    pipeline = STT()

    print(pipeline.listen())
