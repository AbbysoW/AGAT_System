
import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import Linear
from torch import sigmoid

from .transformer import EncoderOnly_Embeddings
from .embaddings_model import EmbaddingsModel

logger = logging.getLogger(__name__)


class BinaryClassifier(nn.Module):
    def __init__(self,
                 num_layers: int,
                 seq_len: int, d_model: int, num_heads: int,
                 d_ff: int,
                 dropout_rate: float = 0.1, 
                 use_positional_encoding: bool = False
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        transformer = EncoderOnly_Embeddings(
            num_layers=num_layers,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            use_positional_encoding=use_positional_encoding
        )
        self.transformer = transformer
        self.linear = Linear(d_model, 1)

    def forward(self, x, mask = None):
        hidden = self.transformer(x, mask)   # [B, seq_len, d_model]
        cls = hidden[:, -1, :]          # берём CLS токен [B, d_model]
        return self.linear(cls).squeeze(-1)  # [B]

class Model:
    def __init__(self):
        # model_state = torch.load("modules/sub_modules/filter_modules/is_important/models/stt_imp_models/models/own_model/model_state.pt")
        # test changes 
        model_state = torch.load(Path(__file__).parent  / "model_state.pt")

        
        cfg = model_state["config"]

        self.model = BinaryClassifier(
            num_layers=cfg["num_layers"],
            seq_len=cfg["seq_len"],
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            use_positional_encoding=True
        )
        self.model.load_state_dict(model_state["model_state_dict"])
        self.model.eval()

        self.embaddin_gmodel = EmbaddingsModel()


    def _make_padding(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [B, seq_len, d_model]
        return (x != 0).all(dim=-1)  # [B, seq_len]

    def __call__(self, x: str):
        embadds = self.embaddin_gmodel(x) # [1, <=seq_len, d_model]
        if embadds.shape[1] < self.model.seq_len:
            pads = torch.zeros([self.model.seq_len - embadds.shape[1],
                                self.model.d_model], 
                                dtype=torch.long, 
                                device='cpu'
                                ).unsqueeze(0)
            embadds = torch.cat([embadds, pads], dim=1)
        embadds = embadds[:, -self.model.seq_len:, :] # [1, seq_len, d_model]

        mask = self._make_padding(embadds)

        y = self.model(embadds, mask)
        return sigmoid(y).item()