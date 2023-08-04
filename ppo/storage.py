import numpy as np

class RolloutStorage():
    def __init__(self, step_nums, env_num, obs_space, act_space):
        self.obs = np.zeros((step_nums, env_num) + obs_space.shape, dtype='float32')
        self.actions = np.zeros((step_nums, env_num) + act_space.shape, dtype='float32')
        self.logprobs = np.zeros((step_nums, env_num), dtype='float32')
        self.rewards = np.zeros((step_nums, env_num), dtype='float32')
        self.dones = np.zeros((step_nums, env_num), dtype='float32')
        self.values = np.zeros((step_nums, env_num), dtype='float32')

        self. step_nums = step_nums
        self.obs_space = obs_space
        self.act_space = act_space
        self.cur_step = 0

    def append(self, obs, action, logprob, reward, done, value):
        self.obs[self.cur_step] = obs
        self.actions[self.cur_step] = action
        self.logprobs[self.cur_step] = logprob
        self.rewards[self.cur_step] = reward
        self.dones[self.cur_step] = done
        self.values[self.cur_step] = value
        self.cur_step = (self.cur_step + 1) % self.step_nums

    def compute_returns(self, value, done, gamma=0.99, gae_lamda=0.95):
        advantages = np.zeros_like(self.rewards)
        lastgaelam = 0
        for t in reversed(range(self.step_nums)):
            if t == self.step_nums - 1:
                nextnonterminal = 1.0 - done
                nextvalues = value.reshape(1, -1)
            else:
                nextnonterminal = 1.0 - self.dones[t+1]
                nextvalues = self.values[t+1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lamda * nextnonterminal * lastgaelam
        returns = advantages + self.values
        self.returns = returns
        self.advantages = advantages
        return advantages, returns
    
    def sample_batch(self, idx):
        b_obs = self.obs.reshape((-1,) + self.obs_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.act_space.shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        return b_obs[idx], b_actions[idx], b_logprobs[idx], b_advantages[idx], b_returns[idx], b_values[idx]