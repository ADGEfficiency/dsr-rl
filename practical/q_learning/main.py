import gym
import tensorflow as tf

from agent import Agent


envs = ['Pendulum-v0', 'CartPole-v1']
env = gym.make(envs[1])


BATCH_SIZE = 64 
EPISODES = 1000
total_steps = EPISODES * 100

with tf.Session() as sess:
    agent = Agent(env, sess, total_steps)

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./logs', graph=sess.graph)
        
    global_rewards = []
    global_step = 0

    for episode in range(10000):
        observation = env.reset()
        done = False
        rewards = []
        while not done:
            global_step += 1 
            if episode % 1000 == 0:
                env.render()
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)

            agent.memory.remember(observation, action, reward, next_observation, done)

            ep_rewards.append(reward)    

            if global_step > 10:
                batch = agent.memory.get_batch(BATCH_SIZE)
                train_info = agent.learn(batch)
                writer.add_summary(train_info['tf_summary'], global_step)

        global_rewards.append(sum(ep_rewards))

        # print('ep {} reward {}'.format(episode, global_rewards[-1]))
        # print('avg lifetime reward {}'.format(sum(global_rewards) / len(global_rewards)))
