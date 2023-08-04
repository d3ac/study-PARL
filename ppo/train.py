import os
os.environ['PARL_BACKEND'] = 'torch'

import parl
import gym
import numpy as np
from model import Model
from agent import Agent
from parl.utils import logger, summary
import argparse

from atari_config import atari_config
from ppo.env_utils import ParallelEnv, LocalEnv
from atari_model import AtariModel



def main():
    # config
    config = atari_config # 离散动作空间
    config['env_num'] = args.env_num
    config['env'] = args.env
    config['seed'] = args.seed
    config['xparl_addr'] = args.xparl_addr
    config['test_every_steps'] = args.test_every_steps
    config['train_total_steps'] = args.train_total_steps
    config['batch_size'] = int(config['env_num'] * config['step_nums'])
    config['num_updates'] = int(config['train_total_steps'] // config['batch_size'])
    # env
    envs = ParallelEnv(config)
    eval_env = LocalEnv(config['env'], test=True)
    obs_space = eval_env.obs_space
    act_space = eval_env.act_space
    # model
    model = AtariModel(obs_space, act_space)
    ppo = parl.algorithms.PPO(
        model, clip_param=config['clip_param'], entropy_coef=config['entropy_coef'],
        initial_lr=config['initial_lr'], continuous_action=config['continuous_action']
    )
    agent = Agent(ppo, config)
    rollout = RolloutStorage(config['step_nums'], config['env_num'], obs_space, act_space)
    # train
    obs, info = envs.reset()
    done = np.zeros(config['env_num'], dtype='float32')
    test_flag = 0
    total_steps = 0

    for update in range(1, config['num_updates'] +1):
        for step in range(config['step_nums']):
            total_steps += 1 * config['env_num']
            value, action, log_prob, _ = agent.sample(obs)
            next_obs, reward, next_done, truncated, info = envs.step(action)
            rollout.append(obs, action, log_prob, reward, done, value.flatten())
            obs , done = next_obs, [next_done[i] or done[i] for i in len(done)]
            # 输出训练信息
            for k in range(config['env_num']):
                if done[k] and "episode" in info[k].keys():
                    logger.info("Training: total steps: {}, episode rewards: {}".format(total_steps, info[k]['episode']['r']))
                    summary.add_scalar("train/episode_reward", info[k]["episode"]["r"], total_steps)
        value = agent.value(obs)
        rollout.compute_returns(value, done)
        value_loss, action_loss, entropy_loss, lr = agent.learn(rollout)
        # test
        if (total_steps + 1) // config['test_every_steps'] >= test_flag:
            pass #!后面还没写






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--env_num', type=int, default=None)
    parser.add_argument('--xparl_addr', type=str, default=None)
    parser.add_argument('--train_total_steps', type=int, default=int(1e7))
    parser.add_argument('--test_every_steps', type=int, default=int(5e3))
    args = parser.parse_args()
    main()