import torch
from torch.nn import Module
from torch_geometric.data import Batch


class DiagOffdiagAvgPooling(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        pair_h = batch["dense_pair_h"]  # shape: B, H, N, N
        N = pair_h.shape[-1]
        diag_sum = torch.sum(torch.diagonal(pair_h, dim1=-2, dim2=-1), dim=2)  # B, H
        diag_avg = diag_sum / N
        if N == 1:
            offdiag_avg = torch.zeros_like(diag_avg)
        else:
            offdiag_avg = (torch.sum(pair_h, dim=[-1, -2]) - diag_sum) / (N * N - N)
        batch.graph_h = torch.cat((diag_avg, offdiag_avg), dim=1)  # B, 2H
        return batch


class DiagOffdiagSumPooling(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch):
        pair_h = batch["dense_pair_h"]  # shape: B, H, N, N
        diag_sum = torch.sum(torch.diagonal(pair_h, dim1=-2, dim2=-1), dim=2)  # B, H
        offdiag_sum = torch.sum(pair_h, dim=[-1, -2]) - diag_sum
        batch.graph_h = torch.cat((diag_sum, offdiag_sum), dim=1)  # B, 2H
        return batch
