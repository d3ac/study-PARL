import parl
import torch
import numpy as np
import torch.nn as nn

class MujocoModel(parl.Model):
    def __init__(self, obs_space, act_space):
        super(MujocoModel, self).__init__()
        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_value = nn.Linear(64, 1)
        self.fc_policy = nn.Linear(64, np.prod(act_space.shape))
        self.fc_pi_std = nn.Parameter(torch.zeros([1, np.prod(act_space.shape)]))
    
    def value(self, obs):
        obs = obs.to(torch.device('cuda')).to(torch.float32)
        out = torch.tanh(self.fc1(obs))
        out = torch.tanh(self.fc2(out))
        value = self.fc_value(out)
        return value
    
    def policy(self, obs):
        obs = obs.to(torch.device('cuda')).to(torch.float32)
        out = torch.tanh(self.fc1(obs))
        out = torch.tanh(self.fc2(out))
        action_mean = self.fc_policy(out)

        action_logstd = self.fc_pi_std
        action_std = torch.exp(action_logstd)
        return action_mean, action_std