import torch
import parl
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(obs_dim, obs_dim * 10)
        self.fc2 = nn.Linear(obs_dim * 10, act_dim)
    
    def forward(self, obs):
        out = F.tanh(self.fc1(obs))
        prob = F.softmax(self.fc2(out), dim=-1)
        return prob