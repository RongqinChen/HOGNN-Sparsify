"""
PPGN with Positional Encoding.
"""

from torch import nn
from torch_geometric.data import Batch

from utils import cfg
from models.dense_input_encoder import DenseInputEncoder
from models.layer import layer_dict
from models.layer.ppgn_update import BlockUpdateLayer
from models.pooling import dense_pooling_dict
from models.output_decoder import output_decoder_dict


class PPGN(nn.Module):
    r"""An implementation of PPGN.
    Args:
        pe_len (int): the length of positional embedding.
        node_encoder (str): node input encoder.
        edge_encoder (str): edge input encoder.
        hidden_dim (int): hidden_dim
        num_layers (int): the number of layers
        mlp_depth (int): the number of layers in each MLP in RPGN
        norm_type (str, optional): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        drop_prob (float, optional): dropout rate.
        pooling (str): Method of graph pooling: avg or sum.
        jumping_knowledge (str, optional): Method of jumping knowledge, last, concat, or LSTM.
        task_type (str): Task type, graph_classification, graph_regression, node_classification.
    """
    def __init__(self) -> None:
        # node_encoder: str = cfg.model.node_encoder
        # edge_encoder: str = cfg.model.edge_encoder
        hidden_dim: int = cfg.model.hidden_dim
        num_layers: int = cfg.model.num_layers
        mlp_depth: int = cfg.model.mlp_depth
        # norm_type: str = cfg.model.norm_type
        # act_type: str = cfg.model.act_type
        pooling: str = cfg.model.pooling
        drop_prob: float = cfg.model.drop_prob
        jumping_knowledge: str = cfg.model.jk_mode
        task_type: str = cfg.model.task_type
        num_tasks: int = cfg.model.num_tasks

        super(PPGN, self).__init__()
        self.dropout = nn.Dropout(drop_prob)

        # 1st part - input encoding
        self.input_encoder = DenseInputEncoder(hidden_dim)

        # 2nd part - aggregation and update
        self.blocks = nn.ModuleList([
            BlockUpdateLayer(hidden_dim, mlp_depth, drop_prob)
            for _ in range(num_layers)
        ])

        # 3rd part - readout
        self.jk = layer_dict['jk'](jumping_knowledge, hidden_dim * 2, num_layers + 1)
        self.dense_pooling = dense_pooling_dict[pooling]()
        # self.sparse_pooling = sparse_pooling_dict[pooling]()

        # 4th part - output decoding
        self.output_decoder = output_decoder_dict[task_type](hidden_dim * 2, num_tasks)

        # 5th part - reset parameters
        self._reset_parameters()

    def alias(self):
        mcfg = cfg.model
        return f"PPGN_L{mcfg.num_layers}D{mcfg.hidden_dim}"

    def _reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.apply(_init_weights)

    def forward(self, batch: Batch) -> Batch:
        batch = self.input_encoder(batch)

        h_list = []
        for block in self.blocks:
            batch = block(batch)
            h_list.append(batch["dense_pair_h"])

        if self.jk is not None:
            batch["dense_pair_h"] = self.jk(h_list)

        batch = self.dense_pooling(batch)
        batch = self.output_decoder(batch)
        return batch
