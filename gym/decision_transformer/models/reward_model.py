import torch
import torch.nn as nn

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

class RewardModel(nn.Module):

    def __init__(self, state_dim, act_dim, device, activation='tanh'):
        super().__init__()

        self.ds = state_dim
        self.da = act_dim
        self.activation = activation

        self.model = nn.Sequential(*gen_net(in_size=self.ds+self.da,
                                                out_size=1, H=256, n_layers=3,
                                                activation=self.activation)).float().to(device)

    def forward(self, states, actions):
        return self.model(torch.cat((states,actions),-1)).squeeze(-1)