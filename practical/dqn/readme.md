
## Setup

I reccomend setting up a virtual Python environment to use for the practical.  This is optional - you can use your system Python.  This is however considered bad practice.

If you use the Anaconda distribution of Python you can
use it to create & manage virtual environments.  You don't need to be in any specific folder - you just need Anaconda on your machine.

Create the environment using Python 3.5

` conda create --name dsrRL python=3.5 `

Activate your environment

` source activate dsrRL `

Now clone the course repo in a folder on your machine

` git clone https://github.com/ADGEfficiency/DSR_RL`

Move into the Q-Learning directory

` cd DSR_RL/practical/dqn/`

Install the packages in requirements.txt

` pip install -r requirements.txt `

Finally we will clone the gym repo - the current pip version is different for one of the environments we are going to
use.  

` git clone https://github.com/openai/gym`

` cd gym `

` python setup.py install`

## Practical work

### Task

I've provided an agent based on the DQN algorithm (i.e. Q-Learning with experience replay and a target network).

The practical is to experiment with changing various hyperparameters to improve learning.  You are welcome to tune any hyperparameter you think will improve learning

Currently supports two of the Open AI gym environments - Pendulum & CartPole.

Hyperparameters to tune include anything in the config_dict:
- learning rate
- observation & reward scaling (both the function and use of history)
- neural network structure (number of layers, nodes per layer)
- target network update frequency
- exploration - ie epsilon decay
- batch size
- discount rate

### Instructions

First activate your Python environment

` source activate dsrRL `

To run an experiment we use main.py

` python main.py `

If you want to change the hyperparameters used in the experiment you can change the `config_dict` dictionary in `main.py`.

If you want to view results of the experiment on Tensorboard you will run a Tensorboard server in a separate terminal session.

In a new terminal

` source activate dsrRL `

` tensorboard --logdir='./results' `

Then open a web browser and go to http://localhost:6006/

## Idea behind this practical
The idea behind this practical is that it's too much to grasp how an entire reinforcement learning system works in an
afternoon.  It's not the goal of the course to give everyone a full understanding of reinforcement learning.

Instead we will focus on a single part of the agent by tuning hyperparameters.  Seeing the effect of hyperparameter changes will hopefully shed light on a small part of how reinforcement learning works.  

We are also training your skill to get up to speed with an exisitng code base.  Since I studied at DSR in Batch 9 I have spent a lot of time looking at RL libraries on GitHub.  

The skill of being given a code base and trying to figure out how it works is valuable.  This skill is trained in this practical.

You will need this skill in industry.  Both to figure out your new companies existing code bases and to understand &
use open source code.
