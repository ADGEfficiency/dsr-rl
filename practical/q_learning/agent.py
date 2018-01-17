from random import random as random_uniform

import gym
import numpy as np

from qfunc import Qfunc
from memory import Memory
from processors import Normalizer


class Agent(object):

    def __init__(self, env, sess, total_steps):
        self.env = env
        self.sess = sess
        self.discount = 0.99
        self.epsilon = EpsilonDecayer(decay_length=total_steps/2)

        if type(env.action_space) == gym.spaces.discrete.Discrete:
            #  the shape of the gym Discrete space is the number of actions
            #  not the shape of a single action array
            obs_space_shape = env.observation_space.shape
            action_space_shape = (1,)
            #  a list of all possible actions
            self.actions = [act for act in range(env.action_space.n)]
        else:
            raise ValueError('Environment not supported')

        self.memory = Memory(obs_space_shape,
                             action_space_shape,
                             size=1000)

        config = {'input_shape': env.observation_space.shape,
                  'output_shape': (len(self.actions),),
                  'layers': [10, 10]}

        self.Qfunc = Qfunc(config)

        self.observation_processor = Normalizer(obs_space_shape[0])
        self.target_processor = Normalizer(1)

    def predict_max_q(self, observations):
        max_q = self.sess.run(self.Qfunc.max_q,
                              {self.Qfunc.observation: observations})
        return max_q.reshape(observations.shape[0], 1)

    def act(self, observation):

        epsilon = self.epsilon()

        if epsilon > random_uniform():
            action = self.env.action_space.sample()

        else:
            observation = observation.reshape((1, *self.env.observation_space.shape))

            observation = self.observation_processor.transform(observation)

            optimal_action_idx = self.sess.run(self.Qfunc.optimal_action_idx,
                                               {self.Qfunc.observation: observation})

            #  index at zero because TF returns an array
            action = self.actions[optimal_action_idx[0]]

        return np.array(action)

    def learn(self, batch):
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminal']
        next_observations = batch['next_observations']

        next_obs_q = self.predict_max_q(next_observations)

        #  if next state is terminal, set the value to zero
        next_obs_q[terminals] = 0

        #  creating a target for Q(s,a) using the Bellman equation
        rewards = rewards.reshape(rewards.shape[0], 1)
        target = rewards + self.discount * next_obs_q
        target = self.target_processor.transform(target)

        #  now the messy part - creating a target array
        #  the target array should be shape=(batch_size, num_actions)
        #  all values = 0 except one - the action being trained (along axis=1)
        targets = np.zeros((target.shape[0], len(self.actions)))

        for k, arr in enumerate(actions):
            idx = self.actions.index(arr)
            targets[k][idx] = target[k]

        observations = self.observation_processor.transform(observations)
        loss, _, summ = self.sess.run([self.Qfunc.loss,
                                       self.Qfunc.train_op,
                                       self.Qfunc.all_summaries],
                                      {self.Qfunc.observation: observations,
                                       self.Qfunc.target: targets})

        return {'loss': loss, 'tf_summary': summ}


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

    def __call__(self): return self.epsilon

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
