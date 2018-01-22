import logging
from random import random as random_uniform

import gym
import numpy as np
import tensorflow as tf

from qfunc import Qfunc
from memory import ReplayMemory
from processors import Normalizer
from utils import make_logger


class Agent(object):
    """
    The learner and decision maker.
    Based on the DQN algorithm - ref Mnih et. al 2015
    i.e. Q-Learning with experience replay & a target network

    All calls to tensorflow are wrapped into methods.


    """
    def __init__(self, env, discount, tau, sess, total_steps):
        self.env = env
        self.discount = discount
        self.tau = 0.01
        self.sess = sess
        self.epsilon_getter = EpsilonDecayer(decay_length=total_steps/2)
        self.target_update_freq = int(total_steps) / 20

        #  we keep track of a learning counter to time target network updates
        self.learn_counter = 0
        self.action_counter = 0 
        self.logger = make_logger('./logs', 'info')

        if type(env.action_space) == gym.spaces.discrete.Discrete:
            #  the shape of the gym Discrete space is the number of actions
            #  not the shape of a single action array
            obs_space_shape = env.observation_space.shape
            action_space_shape = (1,)
            #  a list of all possible actions
            self.actions = [act for act in range(env.action_space.n)]
        else:
            raise ValueError('Environment not supported')

        self.memory = ReplayMemory(obs_space_shape,
                                   action_space_shape,
                                   size=1000)

        config = {'input_shape': env.observation_space.shape,
                  'output_shape': (len(self.actions),),
                  'layers': (10, 10),
                  'learning_rate': 0.0025}

        #  the two approximations of Q(s,a)
        #  use the same config dictionary for both
        self.online = Qfunc(config, scope='online')
        self.target = Qfunc(config, scope='target')

        self.observation_processor = Normalizer(obs_space_shape[0])
        self.target_processor = Normalizer(1)

        self.writer = tf.summary.FileWriter('./tensorboard', graph=self.sess.graph)

    def __repr__(self): return '<class DQN Agent>'

    def predict_target(self, observations):
        """
        Target network is used to predict the maximum discounted expected
        return for the next_observation as experienced by the agent

        args
            observations (np.array)

        returns
            max_q (np.array) shape = (batch_size, 1)
        """
        observations = self.observation_processor.transform(observations)

        fetches = [self.target.max_q, self.target.net_summary]
        feed_dict = {self.target.observation: observations}

        max_q, net_sum = self.sess.run(fetches, feed_dict)
        self.writer.add_summary(net_sum, self.action_counter)

        return max_q.reshape(observations.shape[0], 1)

    def predict_online(self, observation):
        """
        We use our online network to choose actions.

        args
            observation (np.array) a single observation

        returns
            action
        """
        obs = observation.reshape((1, *self.env.observation_space.shape))
        obs = self.observation_processor.transform(obs)

        fetches = [self.online.optimal_action_idx, self.online.net_summary]
        feed_dict = {self.online.observation: obs}
        action_idx, net_sum = self.sess.run(fetches, feed_dict)
        self.writer.add_summary(net_sum, self.action_counter)

        #  index at zero because TF returns an array
        action = self.actions[action_idx[0]]

        logging.debug('predict_online - observation {}'.format(obs))
        logging.debug('predict_online - action_index {}'.format(action_idx))
        logging.debug('predict_online - action {}'.format(action))

        return np.array(action)

    def update_target_network(self):
        """
        Updates the target network weights using the parameter tau

        Relies on the sorted lists of tf.Variables kept in each Qfunc object
        """
        print('updating target network')
        for op, tp in zip(self.online.params, self.target.params):
            print(op.name, tp.name)
            print('copying param {}'.format(op.name))
            tp.assign(tf.multiply(op, self.tau) + tf.multiply(tp, 1 - self.tau))

    def act(self, observation):
        """
        Acting according to epsilon greedy policy

        args
            observation (np.array)

        returns
            action (np.array)
        """
        self.action_counter += 1
        epsilon = self.epsilon_getter.epsilon

        if epsilon > random_uniform():
            action = self.env.action_space.sample()
            logging.debug('acting randomly {}'.format(action))
        else:
            action = self.predict_online(observation)
            logging.debug('acting optimally {}'.format(action))
            
        logging.debug('epsilon is {}'.format(epsilon))
        return np.array(action)

    def learn(self, batch):
        """
        Our agents attempt to make sense of the world.

        A batch sampled using experience replay is used to train the online
        network using targets from the target network.

        args
            batch (dict)

        returns
            train_info (dict)
        """
        self.learn_counter += 1

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminal']
        next_observations = batch['next_observations']

        next_obs_q = self.predict_target(next_observations)

        #  if next state is terminal, set the value to zero
        next_obs_q[terminals] = 0

        #  creating a target for Q(s,a) using the Bellman equation
        rewards = rewards.reshape(rewards.shape[0], 1)
        target = rewards + self.discount * next_obs_q
        target = self.target_processor.transform(target)

        observations = self.observation_processor.transform(observations)

        fetches = [self.online.net_summary,
                   self.online.loss,
                   self.online.train_op,
                   self.online.train_summary]

        feed_dict = {self.online.observation: observations,
                     self.online.target: target}

        net_sum, loss, train_op, train_sum = self.sess.run(fetches, feed_dict) 

        logging.debug('learning - observations {}'.format(observations))
        logging.debug('learning - rewards {}'.format(rewards))
        logging.debug('learning - next_obs_q {}'.format(next_obs_q))
        logging.debug('learning - target {}'.format(target))
        logging.debug('learning - actions {}'.format(actions))
        logging.debug('learning - targets {}'.format(targets))
        logging.debug('learning - loss {}'.format(loss))

        self.writer.add_summary(net_sum, self.learn_counter)
        self.writer.add_summary(train_sum, self.learn_counter)

        if self.learn_counter % self.target_update_freq == 0:
            print('updating target net at LC {}'.format(self.learn_counter))
            self.update_target_network()

        return {'loss': loss}


class EpsilonDecayer(object):
    """
    A class to decay epsilon.  Epsilon is used in e-greedy action selection.

    Initially act totally random, then linear decay to a minimum.

    Two counters are used
        self.count is the total number of steps the object has seen
        self.decay_count is the number of steps in the decay period

    args
        decay_length (int) len of the linear decay period
        init_random (int) num steps to act fully randomly at start
        eps_start (float) initial value of epsilon
        eps_end (float) final value of epsilon
    """

    def __init__(self,
                 decay_length,
                 init_random=0,
                 eps_start=1.0,
                 eps_end=0.1):

        self.decay_length = int(decay_length)
        self.init_random = int(init_random)
        self.min_start = self.init_random + self.decay_length

        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)

        eps_delta = self.eps_start - self.eps_end
        self.coeff = - eps_delta / self.decay_length

        self.reset()

    def __repr__(self): return '<class Epislon Greedy>'

    def reset(self):
        self.count = 0
        self.decay_count = 0

    @property
    def epsilon(self):
        #  move the counter each step
        self.count += 1

        if self.count <= self.init_random:
            self._epsilon = 1.0

        if self.count > self.init_random and self.count <= self.min_start:
            self._epsilon = self.coeff * self.decay_count + self.eps_start
            self.decay_count += 1

        if self.count > self.min_start:
            self._epsilon = self.eps_end

        return float(self._epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)

