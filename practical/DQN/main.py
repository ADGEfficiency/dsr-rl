import logging

import gym
import tensorflow as tf

from agent import Agent
from utils import save_args, make_logger

LOGGER = make_logger('./results/logs.txt', 'info')


def experiment(config):
    """
    A function that runs an experiment.

    args
        config (dict) hyperparameters and experiment setup
    """
    with tf.Session() as sess:

        envs = ['Pendulum-v0', 'CartPole-v0', 'MountainCar-v0']
        env = gym.make(envs[1])

        global_rewards = []
        global_step, episode = 0, 0

        config['env'] = env
        config['env_repr'] = repr(env)
        config['sess'] = sess

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
                if episode % 10 == 0:
                    env.render()
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

            logging.info("""step {:.0f} ep {:.0f}
                         reward {:.1f} avg {:.1f}""".format(global_step, episode,
                                                            ep_rew, avg_reward))

            summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward',
                                                         simple_value=ep_rew)])
            rl_writer.add_summary(summary, episode)
            avg_sum = tf.Summary(value=[tf.Summary.Value(tag='avg_last_100_ep',
                                                         simple_value=avg_reward)])
            rl_writer.add_summary(avg_sum, episode)
            rl_writer.flush()
 
    return config

if __name__ == '__main__':

    config_dict = {'discount': 0.97,
                   'tau': 0.001,
                   'total_steps': 500000,
                   'batch_size': 32,
                   'layers': (50, 50),
                   'learning_rate': 0.0001,
                   'epsilon_decay_fraction': 0.3,
                   'memory_fraction': 0.4,
                   'process_observation': False,
                   'process_target': False}

    output = experiment(config_dict)
