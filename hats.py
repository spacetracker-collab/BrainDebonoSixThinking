
import torch
import torch.nn as nn

class WhiteHat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)
    def forward(self, x):
        return torch.relu(self.net(x))

class RedHat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
    def forward(self, x):
        return self.net(x)

class BlackHat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)
    def forward(self, x):
        return -torch.relu(self.net(x))

class YellowHat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)
    def forward(self, x):
        return torch.relu(self.net(x))

class GreenHat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.net(x)
