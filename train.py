
import torch
from models.debono_net import DeBonoNet
from data.dataset import generate

model = DeBonoNet(16)
X, y = generate()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    out, _ = model(X)
    loss = ((out.mean(dim=1) - y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("Epoch", epoch, "Loss", loss.item())
