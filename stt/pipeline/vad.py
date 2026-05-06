import sounddevice as sd
import numpy as np
import queue

class VAD:
    q = queue.Queue()

    def __init__(self, samplerate: int = 16000, threshold: float = 0.2, silence_duration: float = 1., block_duration: float = 0.1):
        self.samplerate = samplerate
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.block_duration = block_duration
        self.should_stop = False

    def _callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def listen(self):
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            blocksize=int(self.samplerate * self.block_duration),
            callback=self._callback
            ):
            recording = []
            silence_time = 0
            speaking = False

            print("listening...")

            while True:
                data = self.q.get()
                volume = np.sqrt(np.mean(data**2))
                

                if volume > self.threshold:
                    speaking = True
                    silence_time = 0
                    recording.append(data)
                    # print("vol:", volume)
                else:
                    if speaking:
                        silence_time += self.block_duration
                        recording.append(data)
                        # print("vol:", volume)

                        if silence_time >= self.silence_duration:
                            print("Speech ended")
                            break

        audio = np.concatenate(recording, axis=0)
        audio = np.squeeze(audio)

        return audio
    

if __name__ == "__main__":

    vad = VAD()

    audio = vad.listen()
    print(len(audio))