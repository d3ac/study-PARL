import parl
import torch
import numpy as np

class Agent(parl.Agent):
    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm) # 这一句话是必须要的
    
    def sample(self, obs):
        obs = torch.tensor(obs).to(torch.device('cuda:0'))
        prob = self.alg.predict(obs).cpu().detach().numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act
    
    def predict(self, obs):
        obs = torch.tensor(obs).to(torch.device('cuda:0'))
        prob = self.alg.predict(obs).cpu().detach().numpy()
        act = np.argmax(prob)
        return int(act)
    
    def learn(self, obs, act, reward):
        obs = torch.tensor(obs).to(torch.device('cuda:0'))
        act = torch.tensor(act).to(torch.device('cuda:0'))

        running_add = 0
        discounted_reward = np.zeros_like(reward)
        for t in reversed(range(0, len(reward))):
            running_add = running_add * 0.95 + reward[t]
            discounted_reward[t] = running_add
        discounted_reward = (discounted_reward - np.mean(discounted_reward)) / np.std(discounted_reward)

        reward = torch.tensor(discounted_reward, dtype=torch.float32).to(torch.device('cuda:0'))

        loss = self.alg.learn(obs, act, reward)
        return float(loss)