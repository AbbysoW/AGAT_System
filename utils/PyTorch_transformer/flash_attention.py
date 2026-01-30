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
    
    def __init__(self, d_model:int, num_heads:int, block_size:int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.block_size = block_size
        
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
    
    def flash_attention_step(self, qi, kj, vj, o, m, l, mask_block=None): 
        matrix_mult = torch.matmul(qi, kj.transpose(-2, -1))
        # d_k = torch.tensor(self.depth, dtype=q.dtype, device=q.device)
        scaled_logits = matrix_mult / torch.sqrt(self.depth)

        if mask_block is not None:
            scaled_logits += (mask_block * -1e9)

        max_s = scaled_logits.max(dim=-1).values
        m_new = torch.maximum(m, max_s)

        exp_scale_old = torch.exp(m - m_new)
        exp_scale_new = torch.exp(scaled_logits - m_new.unsqueeze(-1))

        l = l * exp_scale_old + exp_scale_new.sum(dim=-1)
        o = o * exp_scale_old.unsqueeze(-1) + torch.matmul(exp_scale_new, vj)

        return o, m_new, l
    

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
        seq_len = q.size(1)

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

        o = torch.zeros([batch_size, self.num_heads, seq_len, self.depth], dtype=q.dtype, device=q.device)
        m = torch.full([batch_size, self.num_heads, seq_len], -1e9, dtype=q.dtype, device=q.device)
        l = torch.zeros([batch_size, self.num_heads, seq_len], dtype=q.dtype, device=q.device)

        #for loop
        for i in range(0, seq_len, self.block_size):
            qi = q[:, :, i:i+self.block_size, :]
            for j in range(0, seq_len, self.block_size):
                kj = k[:, :, j:j+self.block_size, :]
                vj = v[:, :, j:j+self.block_size, :]

                if mask is not None:
                    mask_block = mask[:, :, j:j+self.block_size].unsqueeze(1)
                else:
                    mask_block = None

                o, m, l = self.flash_attention_step(qi, kj, vj, o, m, l, mask_block)

        scaled_attention = o / l.unsqueeze(-1)

        if logger_attention.isEnabledFor(logging.DEBUG):
            pass
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