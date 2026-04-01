
import torch
from debono_net import DeBonoNet
from gnn import SimpleGNN

num_agents = 4
dim = 16

agents = [DeBonoNet(dim) for _ in range(num_agents)]
gnn = SimpleGNN(dim)

states = torch.stack([torch.randn(dim) for _ in range(num_agents)])
adj = torch.ones(num_agents, num_agents) / num_agents

for _ in range(3):
    states = torch.stack([agent(s)[0].squeeze() for agent, s in zip(agents, states)])
    states = gnn(states, adj)

print("Final states:", states)
