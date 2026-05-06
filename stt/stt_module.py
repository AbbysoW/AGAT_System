import re

from .pipeline.models import Whisper
from .pipeline.vad import VAD

class STT:

    _PATTERN = re.compile(r'[A-Za-z0-9\u0400-\u04FF]')
    
    def __init__(self, samplerate: int = 16000):
        self.vad = VAD(samplerate)
        self.should_stop = False
        
        self.medium = Whisper("medium")
    
    def _get_speach_text(self, audio):
        return self.medium(audio)
    
    def _clacuate_speech_probability(self, segments):
        sum = 0
        for seg in segments:
            sum += seg['no_speech_prob']
        return 1 - (sum / len(segments))

    def listen(self) -> str:
        while True:
            try:
                audio = self.vad.listen()
                speach_info = self._get_speach_text(audio)
                print("text:", speach_info)

                if speach_info['text'] and self._PATTERN.search(speach_info['text']):
                    if self._clacuate_speech_probability(speach_info['segments']) > 0.6:
                        return {
                            'text': speach_info['text'],
                            "speaker": "Владелец",
                            'language': speach_info['language']
                        }
                
                print("nothing found")
            except TypeError:
                continue
                

if __name__ == "__main__":

    pipeline = STT()

    print(pipeline.listen())