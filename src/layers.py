import torch
import torch.nn as nn


class SLU(nn.Module):
    def __init__(self, num_parameters=1):
        super(SLU, self).__init__()
        self.k = nn.Parameter(torch.full((num_parameters,), 0.0))

    def forward(self, x):
        A = torch.log(1 + torch.abs(x))
        B = self.k * A.pow(2)
        return torch.where(x > 0, x + B, B - A)
