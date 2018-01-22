import gym
import tensorflow as tf

from agent import Agent


envs = ['Pendulum-v0', 'CartPole-v1']
env = gym.make(envs[1])


BATCH_SIZE = 4  #  low for debug (should be 64)
EPISODES = 1000
DISCOUNT = 0.99
TAU = 0.5 
total_steps = EPISODES * 10

with tf.Session() as sess:
    agent = Agent(env, DISCOUNT, TAU, sess, total_steps)

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./logs', graph=sess.graph)
        
    global_rewards = []
    global_step = 0

    for episode in range(1000):
        observation = env.reset()
        done = False
        rewards = []
        while not done:
            global_step += 1 
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)

            agent.memory.remember(observation, action, reward, next_observation, done)

            rewards.append(reward)    

            batch = agent.memory.get_batch(BATCH_SIZE)
            train_info = agent.learn(batch)

        global_rewards.append(sum(rewards))

        print('ep {} reward {}'.format(episode, global_rewards[-1]))
        print('avg lifetime reward {}'.format(sum(global_rewards) / len(global_rewards)))
