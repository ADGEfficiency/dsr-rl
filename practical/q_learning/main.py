import gym
import tensorflow as tf

from agent import Agent


envs = ['Pendulum-v0', 'CartPole-v1']
env = gym.make(envs[1])

BATCH_SIZE = 32  #  low for debug (should be 64)
DISCOUNT = 0.9
TAU =  0.5
total_steps = 500000

with tf.Session() as sess:
    agent = Agent(env, DISCOUNT, TAU, sess, total_steps)

    sess.run(tf.global_variables_initializer())
    agent.update_target_network()
    rl_writer = tf.summary.FileWriter('./tensorboard/rl')
    global_rewards = []
    global_step = 0
    episode = 0

    while global_step < total_steps:
        observation = env.reset()
        done = False
        rewards = []
        actions = []
        while not done:
            global_step += 1 
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)

            agent.memory.remember(observation, action, reward, next_observation, done)

            rewards.append(reward)    
            actions.append(action)
            batch = agent.memory.get_batch(BATCH_SIZE)
            train_info = agent.learn(batch)
            observation = next_observation

        episode += 1
        global_rewards.append(sum(rewards))

        print('ep {} reward {} lifetime avg {}'.format(episode, 
                                                       global_rewards[-1],
                        sum(global_rewards) / len(global_rewards)))


        ep_rew = sum(rewards)

        summary = tf.Summary(value=[tf.Summary.Value(tag='episode_reward',
                                                     simple_value=ep_rew)])
        rl_writer.add_summary(summary, episode) 
        rl_writer.flush()
