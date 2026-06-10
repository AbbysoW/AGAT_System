import queue
import logging

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)

class VolumeDetector:
    q = queue.Queue()

    def __init__(self, samplerate: int = 16000, threshold: float = 0.2, 
                 silence_duration: float = 3., block_duration: float = 0.1):
        logger.info(f"Initializing VolumeDetector | samplerate={samplerate}Hz | threshold={threshold} | silence_dur={silence_duration}s | block_dur={block_duration}s")
        self.samplerate = samplerate
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.block_duration = block_duration
        self.should_stop = False
        logger.debug("VolumeDetector initialized successfully")

    def _callback(self, indata, frames, time, status):
        """Audio stream callback."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.q.put(indata.copy())

    def __call__(self):
        """Listen for speech using Voice Activity Detection."""
        logger.debug("VolumeDetector listening started")
        try:
            logger.debug(f"Opening audio stream | samplerate={self.samplerate}Hz | blocksize={int(self.samplerate * self.block_duration)}")
            
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                blocksize=int(self.samplerate * self.block_duration),
                callback=self._callback
                ):
                recording = []
                silence_time = 0
                speaking = False
                blocks_processed = 0
                max_volume = 0
                min_volume = float('inf')

                logger.debug("Audio stream opened successfully")

                while True:
                    data = self.q.get()
                    blocks_processed += 1
                    volume = np.sqrt(np.mean(data**2))
                    
                    # Track statistics
                    max_volume = max(max_volume, volume)
                    min_volume = min(min_volume, volume)
                    
                    if volume > self.threshold:
                        if not speaking:
                            logger.debug(f"Speech detected | volume={volume:.3f} > threshold={self.threshold}")
                            speaking = True
                        silence_time = 0
                        recording.append(data)
                        # logger.debug(f"Recording audio block | volume={volume:.3f} | blocks={len(recording)} | duration={len(recording)*self.block_duration:.2f}s")
                    else:
                        if speaking:
                            if silence_time == 0:
                                logger.debug(f"Silence detected | volume={volume:.3f} <= threshold={self.threshold}")
                            silence_time += self.block_duration
                            recording.append(data)
                            # logger.debug(f"Recording silence | silence_duration={silence_time:.2f}s | threshold={self.silence_duration}s")

                            if silence_time >= self.silence_duration:
                                logger.info(f"✓ Speech ended by silence | silence_time={silence_time:.2f}s")
                                break

            audio = np.concatenate(recording, axis=0)
            audio = np.squeeze(audio)
            
            logger.info(f"✓ Audio capture complete | duration={len(audio)/self.samplerate:.2f}s | samples={len(audio)} | blocks={len(recording)}")
            logger.info(f"Volume statistics | min={min_volume:.3f} | max={max_volume:.3f} | threshold={self.threshold}")
            logger.debug(f"Total blocks processed: {blocks_processed}")
            
            return audio
        except Exception as e:
            logger.error(f"✗ VolumeDetector error: {type(e).__name__}: {e}", exc_info=True)
            return
    

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("VolumeDetector Standalone Test")
    logger.info("=" * 60)

    try:
        logger.info("Creating VolumeDetector instance...")
        vad = VolumeDetector()
        
        logger.info("Starting to listen...")
        audio = vad.listen()
        
        logger.info(f"Test completed | audio_length={len(audio)}")
        print(f"Audio length: {len(audio)}")
    except Exception as e:
        logger.error(f"Test failed: {type(e).__name__}: {e}", exc_info=True)
    
    logger.info("=" * 60)