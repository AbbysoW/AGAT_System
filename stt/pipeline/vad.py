import sounddevice as sd
import numpy as np
import queue
import logging

logger = logging.getLogger(__name__)

class VAD:
    q = queue.Queue()

    def __init__(self, samplerate: int = 16000, threshold: float = 0.2, silence_duration: float = 1., block_duration: float = 0.1):
        logger.debug(f"VAD init | samplerate={samplerate} threshold={threshold}")
        self.samplerate = samplerate
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.block_duration = block_duration
        self.should_stop = False

    def _callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def listen(self):
        logger.debug("VAD listening started")
        try:
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                blocksize=int(self.samplerate * self.block_duration),
                callback=self._callback
                ):
                recording = []
                silence_time = 0
                speaking = False

                logger.debug("Audio stream opened")

                while True:
                    data = self.q.get()
                    volume = np.sqrt(np.mean(data**2))
                    

                    if volume > self.threshold:
                        speaking = True
                        silence_time = 0
                        recording.append(data)
                    else:
                        if speaking:
                            silence_time += self.block_duration
                            recording.append(data)

                            if silence_time >= self.silence_duration:
                                logger.debug("Speech ended by silence")
                                break

            audio = np.concatenate(recording, axis=0)
            audio = np.squeeze(audio)
            logger.info(f"Audio captured | duration={len(audio)/self.samplerate:.2f}s size={len(audio)}")
            return audio
        except Exception as e:
            logger.error(f"VAD error: {type(e).__name__}: {e}", exc_info=True)
            raise
    

if __name__ == "__main__":

    vad = VAD()

    audio = vad.listen()
    print(len(audio))