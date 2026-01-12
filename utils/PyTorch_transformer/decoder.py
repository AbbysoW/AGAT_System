import logging

import torch
from torch import nn
from torch.nn import LayerNorm, Dropout

from dot_product_attention import MultiHeadAttention
from feedforward import Feedforward


logger_init = logging.getLogger(f"{__name__}.init")  # Logger for initialization
logger_forward = logging.getLogger(f"{__name__}.forward")  # Logger for forward pass


class Decoder(nn.Module):
    """
    Transformer Decoder Layer.
    
    Processes the target sequence with attention to both itself and the encoder output.
    Each decoder layer consists of:
        1. Masked multi-head self-attention (prevents looking ahead)
        2. Add & Normalize
        3. Multi-head cross-attention to encoder output
        4. Add & Normalize
        5. Feed-forward network
        6. Add & Normalize
    
    Args:
        d_model: Model embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        dropout_rate: Dropout probability for regularization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Sub-layer 1: Masked self-attention (decoder attends to itself)
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 2: Cross-attention (decoder attends to encoder output)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 3: Feed-forward network
        self.feedforward = Feedforward(d_model, d_ff)

        # Dropout layers for regularization
        self.dropout1 = Dropout(p=dropout_rate)
        self.dropout2 = Dropout(p=dropout_rate)
        self.dropout3 = Dropout(p=dropout_rate)

        # Layer normalization layers
        self.layernorm1 = LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = LayerNorm(d_model, eps=1e-6)

        logger_init.info("Decoder initialized")
        logger_init.info("DecoderlConfig: d_model=%s, heads=%s, d_ff=%s, dropout=%s",
                         d_model, num_heads, d_ff, dropout_rate)

    def forward(self, x, enc_output=None, attention_mask=None, padding_mask=None):
        """
        Forward pass through decoder layer.
        
        Args:
            x: Decoder input tensor
            enc_output: Encoder output (for cross-attention)
            training: Whether in training mode
            attention_mask: Causal mask to prevent attending to future positions
            padding_mask: Mask for padded positions
            
        Returns:
            Decoded representation
        """

        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("    Decoder layer input: %s", x.shape)
            if enc_output is not None:
                logger_forward.debug("    Cross-attention from encoder: %s", enc_output.shape)

        # Combine attention mask and padding mask
        inverted_universal_mask = None
        
        if attention_mask is not None and padding_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            universal_mask = attention_mask * padding_mask
            inverted_universal_mask = 1 - universal_mask
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    Using combined mask (attention + padding)")
        
        elif attention_mask is not None:
            universal_mask = attention_mask.unsqueeze(1)
            inverted_universal_mask = 1 - universal_mask
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    Using attention mask only")
        
        elif padding_mask is not None:
            universal_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            inverted_universal_mask = 1 - universal_mask
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    Using padding mask only")
        
        else:
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    No mask applied")

        # Sub-layer 1: Masked self-attention
        attention_output = self.attention(x, x, x, inverted_universal_mask)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layernorm1(x + attention_output)
        
        # If no encoder output provided, use self-attention output
        if enc_output is None:
            enc_output = attention_output
        
        # Sub-layer 2: Cross-attention to encoder
        enc_dec_attention_output = self.enc_dec_attention(attention_output, enc_output, enc_output)
        enc_dec_attention_output = self.dropout2(enc_dec_attention_output)
        enc_dec_attention_output = self.layernorm2(attention_output + enc_dec_attention_output)

        # Sub-layer 3: Feed-forward
        feedforward_output = self.feedforward(enc_dec_attention_output)
        feedforward_output = self.dropout3(feedforward_output)
        output = self.layernorm3(enc_dec_attention_output + feedforward_output)

        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("    Decoder layer output: %s", output.shape)

        return output
    

if __name__ == '__main__':
    x = torch.randn([1, 10, 10])

    decoder = Decoder(10, 2, 5)

    output = decoder.forward(x)

    print(output)