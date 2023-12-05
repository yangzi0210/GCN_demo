import torch.nn as nn
from layers import GraphConvolution
import torch.nn.functional as F


class GCN(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=1433):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(input_dim, 16)
        self.conv2 = GraphConvolution(16, 7)

    def forward(self, adjacency, feature):
        h = self.conv1(adjacency, feature)
        h = F.relu(h)  # (N,1433)->(N,16)
        x = self.conv2(adjacency, h)  # (N,16)->(N,7)
        return x
