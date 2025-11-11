import torch
from torch import nn
from torch_geometric.data import Batch


class BlockMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_depth, drop_prob=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True))
        for _ in range(1, mlp_depth):
            self.norms.append(nn.BatchNorm2d(out_channels))
            self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True))

    def forward(self, inputs):
        out = inputs
        out = self.convs[0](out)
        for idx in range(1, len(self.convs)):
            out = self.norms[idx - 1](out)
            out = self.activation(out)
            out = self.dropout(out)
            out = self.convs[idx](out)

        return out


class BlockMatmulConv(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob=0.0) -> None:
        super().__init__()
        self.mlp1 = BlockMLP(channels, channels, mlp_depth, drop_prob)
        self.mlp2 = BlockMLP(channels, channels, mlp_depth, drop_prob)

    def forward(self, x):  # x: B, H, N, N
        mlp1 = self.mlp1(x)
        mlp2 = self.mlp2(x)
        mult = torch.matmul(mlp1, mlp2)
        out = torch.sqrt(torch.relu(mult)) - torch.sqrt(torch.relu(-mult))
        return out


class BlockUpdateLayer(nn.Module):
    def __init__(self, channels, mlp_depth, drop_prob) -> None:
        super().__init__()
        self.matmul_conv = BlockMatmulConv(channels, mlp_depth, drop_prob)
        self.update = BlockMLP(channels * 2, channels, 2, drop_prob)

    def forward(self, batch: Batch):
        inputs = batch["dense_pair_h"]
        h = self.matmul_conv(inputs)
        h = torch.cat((inputs, h), 1)
        h = self.update(h) + inputs
        batch["dense_pair_h"] = h
        return batch
