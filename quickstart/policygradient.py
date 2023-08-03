import torch
import torch.optim as optim
from torch.distributions import Categorical


class PolicyGradient(object):
    def __init__(self, model, lr):
        """Policy gradient algorithm

        Args:
            model (parl.Model): model defining forward network of policy.
            lr (float): learning rate.

        """
        # checks

        self.model = model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, obs):
        """Predict the probability of actions

        Args:
            obs (torch.tensor): shape of (obs_dim,)

        Returns:
            prob (torch.tensor): shape of (action_dim,)
        """
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward):
        """Update model with policy gradient algorithm

        Args:
            obs (torch.tensor): shape of (batch_size, obs_dim)
            action (torch.tensor): shape of (batch_size, 1)
            reward (torch.tensor): shape of (batch_size, 1)

        Returns:
            loss (torch.tensor): shape of (1)

        """
        prob = self.model(obs)

        log_prob = Categorical(prob).log_prob(action)

        loss = torch.mean(-1 * log_prob * reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss