
import torch
import torch.nn as nn

class BlueHatController(nn.Module):
    def __init__(self, dim, num_hats=5):
        super().__init__()
        self.attn = nn.Linear(dim, num_hats)

    def forward(self, x):
        return torch.softmax(self.attn(x), dim=-1)
