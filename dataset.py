
import torch

def generate(n=1000, dim=16):
    X = torch.randn(n, dim)
    y = (X.sum(dim=1) > 0).float()
    return X, y
