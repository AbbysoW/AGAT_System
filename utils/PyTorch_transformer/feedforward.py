import logging

from torch import nn
from torch.nn import Linear, ReLU


logger_init = logging.getLogger(f"{__name__}.init")  # Logger for initialization


class Feedforward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    A simple two-layer fully connected network applied to each position independently.
    This adds non-linearity and helps the model learn complex patterns.
    
    Architecture: Linear -> ReLU -> Linear
    
    Args:
        d_model: Output dimension (matches model dimension)
        d_ff: Hidden layer dimension (typically 4x d_model)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.dense1 = Linear(d_model, d_ff)   # Expansion layer
        self.dense2 = Linear(d_ff, d_model)   # Projection back to d_model
        self.relu = ReLU()

        logger_init.info("Feedforward initialized")
        logger_init.info("FeedforwardlConfig: d_model=%s, d_ff=%s", 
                         d_model, d_ff)

    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor of same shape as input
        """

        x = self.relu(self.dense1(x))  # Expand and apply ReLU
        output = self.dense2(x)  # Project back to original dimension

        return output
    

if __name__ == '__main__':
    import torch

    x = torch.randn([1, 10, 10])

    feedforward = Feedforward(10, 5)

    output = feedforward.forward(x)

    print(output)
