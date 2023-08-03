import os
os.environ['PARL_BACKEND'] = 'torch'

import parl
import gym
import numpy as np
from model import Model
from agent import Agent
from parl.utils import logger

# ----------------hyperparameters----------------
learning_rate = 1e-2
max_episodes = 1000
# -----------------------------------------------

def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs, info = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)
        obs, reward, done, truncated, info = env.step(action)
        reward_list.append(reward)
        if done or truncated:
            break
    return obs_list, action_list, reward_list

def run_evaluate_episode(agent, eval_episodes=1, render=False):
    if render is True:
        env = gym.make('CartPole-v0', render_mode='human')
    else:
        env = gym.make('CartPole-v0')
    eval_reward = []
    for i in range(eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done or truncated:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def main():
    # env
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    # agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = parl.algorithms.PolicyGradient(model, lr=learning_rate)
    agent = Agent(alg)
    # train loop
    for i in range(max_episodes):
        obs_list, action_list, reward_list = run_train_episode(agent, env)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))
        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = np.array(reward_list)
        agent.learn(batch_obs, batch_action, batch_reward)
        if i % 100 == 0:
            total_reward = run_evaluate_episode(agent, render=True)
            logger.info("Episode {}, Test reward: {}.".format(i, total_reward))


if __name__ == '__main__':
    main()