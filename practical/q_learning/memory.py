from collections import defaultdict, deque, namedtuple
from random import sample

import numpy as np

Experience = namedtuple('experience', ['observation',
                                       'action',
                                       'reward',
                                       'next_observation',
                                       'terminal'])

class ReplayMemory(object):
    """

    """

    def __init__(self, 
                 observation_space_shape, 
                 action_space_shape, 
                 size, 
                 batch_size=64):

        self.size = size
        self.experiences = deque(maxlen=self.size)

        #  use a dict to hold the shapes 
        #  we can use this to eaisly reshape batches of experience
        self.shapes = {'observations': observation_space_shape,
                       'actions': action_space_shape,
                       'rewards': (1,),
                       'next_observations': observation_space_shape,
                       'terminal': (1,)}

    def __repr__(self): return '<class Memory len={}>'.format(len(self))

    def __len__(self): return len(self.experiences)

    def remember(self, observation, action, reward, next_observation, terminal):
        """
         
        """
        #  create an experience named tuple 
        #  add the experience to our deque
        #  the deque automatically keeps our memory at the correct size
        self.experiences.append(Experience(observation, 
                                           action, 
                                           reward, 
                                           next_observation, 
                                           terminal))
    def get_batch(self, batch_size):
        sample_size = min(batch_size, len(self)) 
        batch = sample(self.experiences, sample_size)
        return self.batch_to_dict(batch)
        
    def batch_to_dict(self, batch):
        batch_dict = defaultdict(list)

        for exp in batch:
            batch_dict['observations'].append(exp.observation)
            batch_dict['actions'].append(exp.action)
            batch_dict['rewards'].append(exp.reward)
            batch_dict['next_observations'].append(exp.next_observation)
            batch_dict['terminal'].append(exp.terminal)

        for key, data in batch_dict.items():
            data = np.array(data).reshape(-1, *self.shapes[key])
            batch_dict[key] = data
            assert not np.any(np.isnan(data))

        return batch_dict
