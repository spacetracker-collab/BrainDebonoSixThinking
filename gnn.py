
import torch
import torch.nn as nn

class SimpleGNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.message = nn.Linear(dim, dim)
        self.update = nn.GRUCell(dim, dim)

    def forward(self, states, adj):
        new_states = []
        for i in range(states.size(0)):
            agg = torch.sum(states * adj[i].unsqueeze(-1), dim=0)
            msg = torch.relu(self.message(agg))
            updated = self.update(msg, states[i])
            new_states.append(updated)
        return torch.stack(new_states)
