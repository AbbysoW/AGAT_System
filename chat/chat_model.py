import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import numpy as np

from data_processing import encode, decode

class ChatModel:

    d_model = 764
    seq_len = 10000

    def __init__(self, model_path, embadings):
        self.model = load_model(model_path)

        self.attention_mask =  self.__make_attention_mask()
        self.embadings = embadings

    def __call__(self, seq, padding):

        # seq is [1,2,3,4,5] Not embadings

        seq = seq[self.seq_len:]
        # ЕЩЕ ОБРАБОТКИ SEQ
        embad_seq = encode(seq, self.embadings)

        input = {
            "encoder_input": np.zeros((len(embad_seq), self.seq_len, self.d_model)),  # Dummy input
            "decoder_input": embad_seq,
            "attention_mask": self.attention_mask,
            "decoder_padding_mask": padding
        }
        
        y = self.model.predict(input)
        output = decode(y, self.embadings)

        return output

    def __make_attention_mask(self):
        return np.tril(np.ones((self.seq_len, self.seq_len), dtype=np.int32))