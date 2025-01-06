import torch
from torchmetrics import MeanSquaredLogError


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class MSLE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = MeanSquaredLogError()

    def forward(self, yhat, y):
        return self.loss(yhat, y)
