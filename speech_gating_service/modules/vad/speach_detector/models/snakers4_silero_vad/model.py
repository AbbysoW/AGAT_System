import logging
from pathlib import Path

import torch
from numpy import ndarray


logger = logging.getLogger(__name__)


class Model:
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        torch.set_num_threads(1)
        torch.hub.set_dir(Path(__file__).parent)
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad:v6.2.1', 
            model='silero_vad',
            trust_repo=True
            )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.utils

    def __call__(self, x: ndarray) -> float:
        x = self._preprocess(x)
        logger.debug(f"Preprocessed audio shape: {x.shape}")

        prob = self._get_speech_prob(x)
        
        return prob
    
    def _preprocess(self, x: ndarray):
        audio_np = x.astype('float32')
        logger.debug(f"Preprocessing audio of shape: {audio_np.shape}")
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
            logger.debug(f"Converted stereo to mono audio of shape: {audio_np.shape}")
        audio_tensor = torch.from_numpy(audio_np)
        logger.debug(f"Converted audio to tensor of shape: {audio_tensor.shape}")
        if audio_tensor.abs().max() > 1:
            audio_tensor = audio_tensor / 32768.0
            logger.debug(f"Normalized audio tensor of shape: {audio_tensor.shape}")
        
        return audio_tensor
    
    def _get_speech_prob(self, x: torch.Tensor) -> float:
        chunk_size = int(self.sample_rate * 32e-3)
        logger.debug(f"Calculating speech probability for tensor of shape: {x.shape}")
        logger.debug(f"Chunk size: {chunk_size} | Sample rate: {self.sample_rate} | Sample: 32e-3ms")
        speech_probs = []
        
        for i in range(0, len(x), chunk_size):
            chunk = x[i : i + chunk_size]
            if len(chunk) == chunk_size:
                prob = self.model(chunk, self.sample_rate)
                speech_probs.append(prob.mean().item())

        logger.debug(f"Speech probability calculated. Max: {max(speech_probs):.2f}, Min: {min(speech_probs):.2f}")


        return max(speech_probs) if speech_probs else 0.0