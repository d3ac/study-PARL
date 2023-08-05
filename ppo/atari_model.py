import torch
import parl
from torch import nn
import torch.nn.functional as F

class AtariModel(parl.Model):
    def __init__(self, obs_space, act_space):
        super(AtariModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 512)

        self.fc_pi = nn.Linear(512, act_space.n)
        self.fc_v = nn.Linear(512, 1)
    
    def value(self, obs):
        obs = obs.to(torch.device('cuda'))
        obs = obs / 255.0
        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))
        
        obs = self.flatten(obs)
        obs = F.relu(self.fc(obs))
        v = self.fc_v(obs)
        return v
    
    def policy(self, obs):
        obs = obs.to(torch.device('cuda'))
        obs = obs / 255.0
        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))

        obs = self.flatten(obs)
        obs = F.relu(self.fc(obs))
        logits = self.fc_pi(obs)
        return logits