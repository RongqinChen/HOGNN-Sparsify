from typing import List

import torch
from torch import Tensor, nn

from utils import cfg
from models.act import act_dict
from models.norms import norm_dict


class JumpingKnowledge(torch.nn.Module):
    r""" Adapted from `torch_geometric.nn.models.jumping_knowledge`

    Args:
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
    """
    def __init__(self, mode: str, dim_in: int, num_convs: int) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.dim_in = dim_in
        self.num_convs = num_convs
        if mode == 'last':
            pass
        elif mode == 'cat':
            self.cat_lin = nn.Linear(dim_in * num_convs, dim_in)
            self.cat_norm = norm_dict[cfg.model.post_norm](dim_in)
            self.cat_act = act_dict[cfg.model.post_act]()
            self.cat_dropout = nn.Dropout(cfg.model.post_dropout)
        # elif mode == 'lstm':
        #     layers = cfg.model.jk_lstm_layers
        #     self.lstm = nn.LSTM(dim_in, dim_in, layers, batch_first=True, bidirectional=True)
        #     self.lstm_attn = nn.Linear(layers * 2 * dim_in, cfg.model.jk_lstm_head)
        else:
            raise ValueError(f"JK mode `{mode}` is invalid.")

    def forward(self, xs: List[Tensor]) -> Tensor:
        r"""Forward pass.

        Args:
            xs (List[torch.Tensor]): List containing the layer-wise representations.
        """
        if self.mode == 'last':
            return xs[-1]
        elif self.mode == 'cat':
            h = torch.cat(xs, dim=-1)  # [*, num_convs * num_channels]
            h = self.cat_lin(h)
            h = self.cat_norm(h)
            h = self.cat_act(h)
            h = self.cat_dropout(h)
            return h
        # elif self.mode == 'lstm':
        #     x = torch.stack(xs, dim=1)  # [num_nodes, num_convs, num_channels]
        #     alpha, _ = self.lstm(x)
        #     alpha = self.lstm_attn(alpha)  # [num_nodes, num_convs, heads]
        #     alpha = torch.softmax(alpha, dim=1)
        #     alpha = alpha.tile((1, 1, x.shape[2] // alpha.shape[2]))
        #     h = (x * alpha).sum(dim=1)  # [num_nodes, num_channels]
        #     return h
        else:
            pass

    def __repr__(self) -> str:
        if self.mode == 'cat':
            return (f'{self.__class__.__name__}({self.mode}, dim_in={self.dim_in}, num_convs={self.num_convs})')
        # if self.mode == 'lstm':
        #     return (f'{self.__class__.__name__}({self.mode}, layers={cfg.model.jk_lstm_layers})')
        return f'{self.__class__.__name__}({self.mode})'
