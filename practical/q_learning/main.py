import logging

import gym
import tensorflow as tf

from agent import Agent
from utils import save_args, make_logger


logger = make_logger('./results/logs.txt', 'info')

with tf.Session() as sess:

    envs = ['Pendulum-v0', 'CartPole-v1']
    env = gym.make(envs[1])

    config = {'env': env,
              'env_repr': repr(env),
              'discount': 0.9,
              'tau': 0.001,
              'sess': sess,
              'total_steps': 45000,
              'batch_size': 64,
              'layers': (64, 64, 64),
              'learning_rate': 0.001,
              'epsilon_decay_fraction': 0.5,
              'memory_fraction': 0.25,
              'process_observation': False,
              'process_target': False}

    global_rewards = []
    global_step, episode = 0, 0

    agent = Agent(**config)

    rl_writer = tf.summary.FileWriter('./results/rl')
    save_args(config, 'results/args.txt')

    while global_step < config['total_steps']:
        episode += 1
        done = False
        rewards, actions = [], []
        observation = env.reset()

        while not done:
            global_step += 1

            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, next_observation, done)
            train_info = agent.learn()

            rewards.append(reward)
            actions.append(action)

            observation = next_observation

        ep_rew = sum(rewards)
        global_rewards.append(ep_rew)
        avg_reward = sum(global_rewards[-100:]) / len(global_rewards[-100:])

        logging.info('step {:.0f} ep {:.0f} reward {:.1f} avg {:.1f}'.format(global_step,
                                                               episode,
                                                               ep_rew,
                                                               avg_reward))

        summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward',
                                                     simple_value=ep_rew)])
        rl_writer.add_summary(summary, episode)
        avg_sum  = tf.Summary(value=[tf.Summary.Value(tag='avg_last_100_ep',
                                                     simple_value=avg_reward)])
        rl_writer.add_summary(avg_sum, episode)
        rl_writer.flush()
