import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

load_dotenv()


class Model:

    def __init__(self):
        self.model_name = 'boltuix/bert-lite'

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=Path(__file__).parent,
            )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=Path(__file__).parent,
            add_pooling_layer=False,
            ignore_mismatched_sizes=True,
        )
        self.model.eval()

    def __call__(self, x):
        encoded_input = self.tokenizer(
            x, 
            padding=True, 
            truncation=True,
            return_tensors='pt')
        if encoded_input['input_ids'].shape[1] > 512:
            for key in encoded_input:
                encoded_input[key] = encoded_input[key][:, -512:]
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return model_output.last_hidden_state

if  __name__ ==  '__main__':
    model = Model()
    sentence = "Hello, my dog is cute"
    embedding = model(sentence)
    print(embedding.shape)
