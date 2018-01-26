import gym
import tensorflow as tf

from agent import Agent
from utils import save_args


with tf.Session() as sess:

    config = {'env': env,
              'env_repr': repr(env),
              'discount': 0.9,
              'tau': 0.5,  #  not being used
              'sess': sess,
              'total_steps': 500000,
              'batch_size': 64,
              'layers': (10, 10, 10),
              'learning_rate': 0.0001}

    global_rewards = []
    global_step, episode = 0, 0

    envs = ['Pendulum-v0', 'CartPole-v1']
    env = gym.make(envs[0])
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
            agent.memory.remember(observation, action, reward, next_observation, done)
            train_info = agent.learn()

            rewards.append(reward)
            actions.append(action)
            observation = next_observation

        ep_rew = sum(rewards)
        global_rewards.append(ep_rew)
        avg_reward = sum(global_rewards) / len(global_rewards)

        logging.info('step {} ep {} reward {} lifetime avg {0:.2f}'.format(global_step,
                                                               episode,
                                                               ep_rew,
                                                               avg_reward))

        summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward',
                                                     simple_value=ep_rew)])
        rl_writer.add_summary(summary, episode)
        rl_writer.flush()
