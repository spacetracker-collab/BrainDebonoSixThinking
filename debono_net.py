
import torch
import torch.nn as nn
from hats import *
from controller import BlueHatController

class DeBonoNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.white = WhiteHat(dim)
        self.red = RedHat(dim)
        self.black = BlackHat(dim)
        self.yellow = YellowHat(dim)
        self.green = GreenHat(dim)
        self.controller = BlueHatController(dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = [
            self.white(x),
            self.red(x),
            self.black(x),
            self.yellow(x),
            self.green(x)
        ]

        weights = self.controller(x)
        stacked = torch.stack(h, dim=1)  # (batch, 5, dim)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        return self.output(fused), weights
