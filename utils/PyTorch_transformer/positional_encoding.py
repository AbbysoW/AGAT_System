import numpy as np
from torch import from_numpy


def positional_encoding(d_model: int, seq_len: int):
    """
    Generates sinusoidal positional encodings for the Transformer model.
    
    This function creates position-dependent patterns that allow the model to understand
    the order of tokens in a sequence, since attention mechanism itself is permutation-invariant.
    
    Args:
        d_model: Embedding dimension of the model
        seq_len: Maximum sequence length
        
    Returns:
        Tensor of shape (1, seq_len, d_model) containing positional encodings
        
    Mathematical formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    # Create position indices [0, 1, 2, ..., seq_len-1]
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))

    # Apply sin to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # Add batch dimension
    pos_encoding = angle_rads[np.newaxis, ...]

    return from_numpy(pos_encoding).float()

if __name__ == '__main__':

    positional_encodings = positional_encoding(5, 10)

    print(positional_encodings)

