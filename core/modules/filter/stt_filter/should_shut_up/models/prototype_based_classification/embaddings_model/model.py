import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()


class Model:

    def __init__(self):
        self.model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

        self.model = SentenceTransformer(
            self.model_name, 
            cache_folder=Path(__file__).parent
            )
        self.model.eval()

    def __call__(self, x):
        return torch.tensor(self.model.encode(x), dtype=torch.float32)

if  __name__ ==  '__main__':
    model = Model()
    sentence = "Hello, my dog is cute"
    embedding = model(sentence)
    print(embedding.shape)
    print(type(embedding))
