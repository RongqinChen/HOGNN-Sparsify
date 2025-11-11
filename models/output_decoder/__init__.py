"""
Different output decoders for different datasets/tasks.
"""

from torch import nn
from torch_geometric.data import Batch
from utils import cfg


class GraphRegression(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(GraphRegression, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2), nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels // 4), nn.ReLU(),
            nn.Linear(in_channels // 4, out_channels)
        )

    def forward(self, batch: Batch) -> Batch:
        batch["graph_pred"] = self.regressor(batch["graph_h"])
        return batch


class GraphClassification(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(GraphClassification, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2), nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels // 4), nn.ReLU(),
            nn.Linear(in_channels // 4, out_channels)
        )

    def forward(self, batch: Batch) -> Batch:
        batch["graph_pred"] = self.regressor(batch["graph_h"])
        return batch


class MLPGraphHead(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(MLPGraphHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(cfg.model.output_drop_prob),
            nn.Linear(in_channels, in_channels), nn.GELU(),
            nn.Dropout(cfg.model.output_drop_prob),
            nn.Linear(in_channels, in_channels), nn.GELU(),
            nn.Dropout(cfg.model.output_drop_prob),
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, batch: Batch) -> Batch:
        batch["graph_pred"] = self.mlp(batch["graph_h"])
        return batch


class NodeClassification(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(NodeClassification, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2), nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels)
        )

    def forward(self, batch: Batch) -> Batch:
        batch["node_pred"] = self.classifier(batch["node_h"])
        return batch


class NodeRegression(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(NodeRegression, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2), nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels)
        )

    def forward(self, batch: Batch) -> Batch:
        batch["node_pred"] = self.regressor(batch["node_h"])
        return batch


output_decoder_dict = {
    "graph_regression": GraphRegression,
    "graph_classification": GraphClassification,
    "mlpgraphhead": MLPGraphHead,
    "node_classification": NodeClassification,
    "node_regression": NodeRegression,
}

__all__ = ["output_decoder_dict"]
