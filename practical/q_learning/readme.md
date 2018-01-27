## Setup

Optional but reccomended - setup a virtual Python environment.  If you use tthe Anaconda distribution of Python you can
use conda to create a virtual environment as follows

` conda create --name dsrRL python=3.5 `

Then activate your environment

` source activate dsrRL `

Install the packages in requirements.txt

` pip install requirements.txt `

Finally we will clone the gym repo - the current pip version is different for one of the environments we are going to
use.  Make sure you have activated your virtual environment first!  

` git clone https://github.com/openai/gym`

` cd gym `

` python setup.py install`


## Practical
The idea behind this practical is that it's too much to grasp how an entire reinforcement learning system works in an
afternoon.  Instead we will focus on a single part of the agent by tuning hyperparameters.

Seeing the effect of hyperparameter changes will hopefully shed light on the entire learning process.

Since I studied at DSR in Batch 9 I have spent a lot of time looking at RL libraries on GitHub.  The skill of being
given a code base and trying to figure out how it works is valuable.  This skill is trained in this practical.

You will need this skill in industry.  Both to figure out your new companies existing code bases and to understand &
use open source code.

## Work
I've provided an agent based on the DQN algorithm (i.e. Q-Learning with experience replay and a target network).

The practical is to experiment with changing various hyperparameters to improve learning.  You are welcome to tune any
hyperparameter you think will improve learning

Currently supports two of the Open AI gym environments - Pendulum & CartPole.

Hyperparameters to tune:
- learning rate
- observation & reward scaling (both the function and use of history)
- neural network strucutre (number of layers, nodes per layer)
- target network update frequency
- exploration - ie epsilon decay
- batch size
- discount rate
