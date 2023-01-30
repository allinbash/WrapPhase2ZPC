import torch
from torch import nn
from torch.nn import L1Loss
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.nn import CTCLoss
from torch.nn import NLLLoss
from torch.nn import KLDivLoss
from torch.nn import BCELoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import MarginRankingLoss
from torch.nn import HingeEmbeddingLoss
from torch.nn import MultiLabelMarginLoss
from torch.nn import HuberLoss
from torch.nn import SmoothL1Loss
from torch.nn import SoftMarginLoss
from torch.nn import MultiLabelSoftMarginLoss
from torch.nn import CosineEmbeddingLoss
from torch.nn import MultiMarginLoss
from torch.nn import TripletMarginLoss
from torch.nn import TripletMarginWithDistanceLoss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-6

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)
