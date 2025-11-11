import torch
from torch.nn import Module
from torch_geometric.data import Batch


class AdaptiveDiagOffdiagAvgPooling(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        dense_node_mask = batch["dense_node_mask"]
        num_nodes = dense_node_mask.sum(dim=1, keepdim=True)
        dense_node_mask = dense_node_mask.unsqueeze(1)
        dense_node_mask = dense_node_mask.unsqueeze(-1) * dense_node_mask.unsqueeze(2)
        num_offdiag = num_nodes ** 2 - num_nodes

        pair_h = batch["dense_pair_h"] * dense_node_mask  # shape: B, H, N, N
        batchsize = pair_h.shape[-1]
        diag_sum = torch.sum(torch.diagonal(pair_h, dim1=-2, dim2=-1), dim=2)  # B, H
        diag_avg = diag_sum / num_nodes
        if batchsize == 1:
            offdiag_avg = torch.zeros_like(diag_avg)
        else:
            offdiag_avg = (torch.sum(pair_h, dim=[-1, -2]) - diag_sum) / num_offdiag
        batch["graph_h"] = torch.cat((diag_avg, offdiag_avg), dim=1)  # B, 2H
        return batch


class AdaptiveDiagOffdiagSumPooling(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        dense_node_mask = batch["dense_node_mask"]
        # num_nodes = dense_node_mask.sum(dim=1)
        dense_node_mask = dense_node_mask.unsqueeze(1)
        dense_node_mask = dense_node_mask.unsqueeze(-1) * dense_node_mask.unsqueeze(2)

        pair_h = batch["dense_pair_h"] * dense_node_mask  # shape: B, H, N, N
        diag_sum = torch.sum(torch.diagonal(pair_h, dim1=-2, dim2=-1), dim=2)  # B, H
        offdiag_sum = torch.sum(pair_h, dim=[-1, -2]) - diag_sum
        batch["graph_h"] = torch.cat((diag_sum, offdiag_sum), dim=1)  # B, 2H
        return batch
