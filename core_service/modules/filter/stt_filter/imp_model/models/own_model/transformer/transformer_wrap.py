# НАДО переписать класс трансформера (Частично) 
# Теперь будет несколько кслассо специализирующихся под разные архитектуры трансформера
# Теперь класс гне будет учитывать все возможные варианты а бкдет требовать строгого соблюдения инпута
# Инпут будет листом интов а не эмбэдингов как в изначальной задумке (хотя отдельные классы для этого тоже можно реализовать)
import torch
from torch import nn
from torch.nn import Dropout

from .encoder import Encoder
from .positional_encoding import positional_encoding

# ENCODER ONLY - EMBADDINGS
class EncoderOnly_Embeddings(nn.Module):

    def __init__(self,
                 num_layers: int,
                 seq_len: int, d_model: int, num_heads: int,
                 d_ff: int,
                 dropout_rate: float = 0.1, 
                 use_positional_encoding: bool = False):
        super().__init__()

        self.num_layers = num_layers
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.use_positional_encoding = use_positional_encoding

        self.dropout = Dropout(dropout_rate)

        # Positional Encoding
        if use_positional_encoding:
            pos_encoding = positional_encoding(d_model, seq_len)
            self.register_buffer("pos_encoding", pos_encoding)

        # Layers
        self.layers = nn.ModuleList([Encoder(d_model, num_heads, d_ff, dropout_rate**0.5) 
                       for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        if self.use_positional_encoding:
            x += self.pos_encoding[:, :self.seq_len, :]

        x = self.dropout(x)
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        for i, layer in enumerate(self.layers):
            x = layer(x, padding_mask)

        return x

