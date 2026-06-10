import logging

import torch
from torch import nn
from torch.nn import LayerNorm, Dropout

# from dot_product_attention import MultiHeadAttention
from .pytorch_attention import MultiHeadAttention
from .feedforward import Feedforward


logger_init = logging.getLogger(f"{__name__}.init")  # Logger for initialization
logger_forward = logging.getLogger(f"{__name__}.forward")  # Logger for forward pass


class Encoder(nn.Module):
    """
    Transformer Encoder Layer.
    
    Processes the input sequence through self-attention and feed-forward networks.
    Each encoder layer consists of:
        1. Multi-head self-attention
        2. Add & Normalize (residual connection + layer normalization)
        3. Feed-forward network
        4. Add & Normalize (residual connection + layer normalization)
    
    Args:
        d_model: Model embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        dropout_rate: Dropout probability for regularization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Sub-layer 1: Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 2: Feed-forward network
        self.feedforward = Feedforward(d_model, d_ff)

        # Layer normalization for stabilizing training
        self.layernorm1 = LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = LayerNorm(d_model, eps=1e-6)

        # Dropout for regularization
        self.dropout1 = Dropout(p=dropout_rate)
        self.dropout2 = Dropout(p=dropout_rate)

        logger_init.info("Encoder initialized")
        logger_init.info("EncoderlConfig: d_model=%s, heads=%s, d_ff=%s, dropout=%s", # Поменять стиль вывода 
                         d_model, num_heads, d_ff, dropout_rate)

    def forward(self, x, padding_mask=None):
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor
            training: Whether in training mode (affects dropout)
            padding_mask: Mask for padded positions (1 = valid, 0 = padding)
            
        Returns:
            Encoded representation
        """

        if logger_forward.isEnabledFor(logging.DEBUG):
            mask_status = "with padding mask" if padding_mask is not None else "no mask"
            logger_forward.debug("    Encoder layer input: %s (%s)", x.shape, mask_status)


        # Sub-layer 1: Self-attention
        attention_output = self.attention(x, x, x, padding_mask)
        attention_output = self.dropout1(attention_output)
        # Residual connection + layer normalization
        attention_output = self.layernorm1(x + attention_output)

        # Sub-layer 2: Feed-forward
        feedforward_output = self.feedforward(attention_output)
        feedforward_output = self.dropout2(feedforward_output)
        # Residual connection + layer normalization
        output = self.layernorm2(attention_output + feedforward_output)

        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("    Encoder layer output: %s", output.shape)

        return output
    

if __name__ == '__main__':
    x = torch.randn([1, 10, 10])

    encoder = Encoder(10, 2, 5)

    output = encoder.forward(x)

    print(output)