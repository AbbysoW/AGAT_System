import logging

import torch
from torch import nn
from torch.nn import Linear
from torch.nn import functional as fun


logger_init = logging.getLogger(f"{__name__}.init")  # Logger for initialization
logger_attention = logging.getLogger(f"{__name__}.attention")  # Logger for MultiHeadAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism - the core component of the Transformer architecture.
    
    This layer splits the input into multiple attention heads, allowing the model to jointly
    attend to information from different representation subspaces at different positions.
    
    Args:
        d_model: Total dimension of the model (must be divisible by num_heads)
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0

        # Dimension of each attention head
        self.depth = d_model // num_heads

        # Linear projections for Query, Key, Value
        self.wq = Linear(d_model, d_model)  # Query projection
        self.wk = Linear(d_model, d_model)  # Key projection
        self.wv = Linear(d_model, d_model)  # Value projection
        
        # Final linear projection after concatenating all heads
        self.dense = Linear(d_model, d_model)

        logger_init.info("MultiHeadAttention initialized")
        logger_init.info("MultiHeadAttentionlConfig: d_model=%s, heads=%s", 
                         d_model, num_heads)

    def split_heads(self, x, batch_size: int):
        """
        Splits the last dimension into (num_heads, depth) and transposes for parallel processing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Size of the batch
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # Reshape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, depth)
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)


        # Transpose: (batch_size, seq_len, num_heads, depth) -> (batch_size, num_heads, seq_len, depth)
        return x.permute(0, 2, 1, 3)

    def dot_product_attention(self, q, k, v, mask=None):
        """
        Computes scaled dot-product attention.
        
        This calculates how much each word in a sequence should "attend to" every other word.
        The attention weights determine which parts of the input are most relevant.
        
        Args:
            q: Query matrix of shape (..., seq_len_q, depth)
            k: Key matrix of shape (..., seq_len_k, depth)
            v: Value matrix of shape (..., seq_len_v, depth)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            output: Attention-weighted values
            attention_weights: Attention probability distribution
            
        Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        """
        # Calculate attention scores: Q * K^T
        matrix_mult = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale by square root of key dimension (prevents softmax saturation)
        d_k = torch.tensor(k.size(-1), device=k.device, dtype=torch.float32)
        scaled_logits = matrix_mult / torch.sqrt(d_k)

        # Apply mask (if provided) by adding large negative values to masked positions
        if mask is not None:
            scaled_logits += (mask * -1e9)

        # Apply softmax to get attention probabilities
        attention_weights = fun.softmax(scaled_logits, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query input
            k: Key input
            v: Value input
            mask: Optional attention mask
            
        Returns:
            Output after multi-head attention and linear projection
        """
        batch_size = q.size(0)

        if logger_attention.isEnabledFor(logging.DEBUG):
            logger_attention.debug("      MHA: Q=%s, K=%s, V=%s", 
                                  q.shape, k.shape, v.shape)
            if mask is not None:
                logger_attention.debug("      Mask applied: %s", mask.shape)

        # Linear projections for Q, K, V
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Apply attention function
        scaled_attention, attention_weights = self.dot_product_attention(q, k, v, mask)

        if logger_attention.isEnabledFor(logging.DEBUG):
            # Average attention weight for all heads
            mean_attention = attention_weights.mean().item()
            logger_attention.debug(
                "      Attention weights: mean=%.4f, shape=%s",
                mean_attention,
                tuple(attention_weights.shape)
            )   

        # Transpose back: (batch_size, num_heads, seq_len, depth) -> (batch_size, seq_len, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        
        # Concatenate heads: (batch_size, seq_len, num_heads, depth) -> (batch_size, seq_len, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.dense(concat_attention)

        return output


if __name__ == '__main__':
    attention = MultiHeadAttention(10, 2)

    q = torch.randn([3, 10, 10])
    k = torch.randn([3, 10, 10])
    v = torch.randn([3, 10, 10])

    attention_mask = 1 - torch.tril(torch.ones(10, 10))

    print(attention_mask)

    output = attention.forward(q, k, v, mask=attention_mask)

    print(output)