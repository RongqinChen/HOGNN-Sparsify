from torch.nn import Module
from torch_geometric.data import Batch
from torch_geometric.utils import scatter


class GraphAvgPooling(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        node_batch = batch.batch if hasattr(batch, 'batch') else batch['node'].batch
        batch.graph_h = scatter(batch.node_h, node_batch, dim=0, dim_size=batch.num_graphs, reduce='mean')
        return batch


class GraphSumPooling(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        node_batch = batch.batch if hasattr(batch, 'batch') else batch['node'].batch
        batch.graph_h = scatter(batch.node_h, node_batch, dim=0, dim_size=batch.num_graphs, reduce='sum')
        return batch


class GraphMaxPooling(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        node_batch = batch.batch if hasattr(batch, 'batch') else batch['node'].batch
        batch.graph_h = scatter(batch.node_h, node_batch, dim=0, dim_size=batch.num_graphs, reduce='max')
        return batch
