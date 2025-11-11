import torch
import torch_sparse
from torch import nn
from torch_scatter import scatter_add
from utils import cfg
from models.input_encoder import edge_encoder_dict, node_encoder_dict
from models.output_decoder import output_decoder_dict


class SPPGN1Layer(nn.Module):
    def __init__(self, hidden_dim, use_sqrt, drop_prob):
        super().__init__()
        self.use_sqrt = use_sqrt
        hdim = hidden_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.BatchNorm1d(hdim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hdim, hdim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.BatchNorm1d(hdim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hdim, hdim),
        )
        self.upd = nn.Sequential(
            nn.Linear(hdim * 2, hdim),
            nn.BatchNorm1d(hdim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hdim, hdim),
        )

    def forward(self, data):
        idx0, idx1, idx2 = data["3tuple_index"]
        x2 = data["pair_h"]
        x2_1 = self.mlp1(x2)
        x2_2 = self.mlp2(x2)

        x3 = x2_1[idx1] * x2_2[idx2]
        x3_agg = scatter_add(x3, idx0, dim=0, dim_size=x2.size(0))
        if self.use_sqrt:
            x3_agg = torch.sqrt(torch.relu(x3_agg)) - torch.sqrt(torch.relu(-x3_agg))

        h2 = torch.cat([x2, x3_agg], dim=-1)
        data["pair_h"] = self.upd(h2) + x2
        return data

    def extra_repr(self) -> str:
        return f"use_sqrt={self.use_sqrt}"


class SPPGN1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hdim = cfg.model.hidden_dim
        num_layers = cfg.model.num_layers
        use_sqrt = True
        drop_prob = cfg.model.drop_prob

        self.init_layer = InitLayer()
        self.layers = nn.ModuleList(
            [SPPGN1Layer(hdim, use_sqrt, drop_prob) for _ in range(num_layers)]
        )
        self.pooling = cfg.model.pooling
        output_decoder = output_decoder_dict[cfg.model.task_type]
        self.output_decoder = output_decoder(hdim * 2, cfg.model.num_tasks)
        self._reset_parameters()

    def _reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.apply(_init_weights)

    def forward(self, data):
        data = self.init_layer(data)
        for layer in self.layers:
            data = layer(data)

        pair_h = data["pair_h"]
        loop_idx = data["loop_idx"]
        agg2 = scatter_add(pair_h, data["pair_x_batch"], dim=0, dim_size=data.num_graphs)
        agg1 = scatter_add(pair_h[loop_idx], data["batch"], dim=0, dim_size=data.num_graphs)
        agg2 = agg2 - agg1

        graph_node_sizes = torch.diff(data.ptr).view((-1, 1))
        if self.pooling in {"avg", "mean"}:
            graph_pair_sizes = torch.diff(data["pair_x_ptr"]).view((-1, 1))
            diff_sizes = graph_pair_sizes - graph_node_sizes
            diff_sizes = torch.clip(diff_sizes, min=1)
            agg2 = agg2 / diff_sizes
            agg1 = agg1 / graph_node_sizes
        elif self.pooling == "sum_avg":
            agg2 = agg2 / graph_node_sizes

        data["graph_h"] = torch.cat((agg1, agg2), dim=-1)
        data = self.output_decoder(data)
        return data


class InitLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        self.node_encoder = node_encoder_dict[cfg.model.node_encoder](hidden_dim)
        self.edge_encoder = edge_encoder_dict[cfg.model.edge_encoder](hidden_dim)
        self.loop_lin = nn.Linear(cfg.dataset.poly_dim, hidden_dim, False)
        self.pair_lin = nn.Linear(cfg.dataset.poly_dim, hidden_dim, False)
        if "conn" in cfg.model.loop_encoder:
            conn_dim = int(cfg.model.loop_encoder.split("+")[0][4:])
            self.loop_conn = nn.Linear(conn_dim, hidden_dim, False)
            self.pair_conn = nn.Linear(conn_dim, hidden_dim, False)

    def forward(self, data):
        data = self.node_encoder(data)
        data = self.edge_encoder(data)
        poly_dim = cfg.dataset.poly_dim

        loop_x = data["loop_x"]
        pair_x = data["pair_x"]
        if "conn" not in cfg.model.loop_encoder:
            node_val = self.loop_lin(loop_x)
            pair_val = self.pair_lin(pair_x)
        else:
            conn_dim = int(cfg.model.loop_encoder.split("+")[0][4:])
            loop_h = self.loop_lin(loop_x[:, conn_dim:])
            pair_h = self.pair_lin(pair_x[:, conn_dim:])
            loop_conn = self.loop_conn(loop_x[:, :conn_dim]) / conn_dim * poly_dim
            pair_conn = self.pair_conn(pair_x[:, :conn_dim]) / conn_dim * poly_dim
            node_val = loop_h + loop_conn
            pair_val = pair_h + pair_conn

        node_rng = torch.arange(data.num_nodes, device=node_val.device)
        loop_idx = torch.stack([node_rng, node_rng])

        node_val = data["node_h"] + node_val
        pair_idx = data["pair_index"]
        edge_idx = data["edge_index"]
        edge_val = data["edge_h"]
        _, pair_h = torch_sparse.coalesce(
            torch.cat([pair_idx, edge_idx, loop_idx], dim=1),
            torch.cat([pair_val, edge_val, node_val], dim=0),
            data.num_nodes, data.num_nodes, op="add",
        )
        data["pair_h"] = pair_h
        return data
