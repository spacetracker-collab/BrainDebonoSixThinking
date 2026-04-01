
import torch
from models.debono_net import DeBonoNet
from models.gnn import SimpleGNN

num_agents = 4
dim = 16

agents = [DeBonoNet(dim) for _ in range(num_agents)]
gnn = SimpleGNN(dim)

states = torch.stack([torch.randn(dim) for _ in range(num_agents)])

adj = torch.ones(num_agents, num_agents) / num_agents

for step in range(3):
    new_states = []
    for i, agent in enumerate(agents):
        s, _ = agent(states[i])
        new_states.append(s.squeeze())
    states = torch.stack(new_states)
    states = gnn(states, adj)

print("Final states:", states)
