---?image=assets/images/section2/learn_plan_act.png&size=auto 80%

![fig](assets/images/section_1/func_approx.png)

---
# a glance at reinforcement learning

## Adam Green
## [adam.green@adgefficiency.com](adam.green@adgefficiency.com)
## [adgefficiency.com](http://adgefficiency.com)
---
### Course Materials
All course materials are in the GitHub repo DSR_RL.

The materials are
- lecture notes hosted on GitPages
- a collection of useful machine learning & reinforcement learning literature
- practical work, consisting of a collection of scripts to run DQN on Cartpole and some additional Python tips & tricks
---
### Agenda

***today - morning***

one - background & terminology
two - introduction to reinforcement learning
three - value functions

***today - afternoon***

DQN practical

***tomorrow - morning***

four - improvements to DQN
five - policy gradients
six - practical concerns
seven - a quick look at the state of the art

---

### About Me

**Education**

2006 - 2011 B.Eng Chemical Engineering

2011 - 2012 MSc Advanced Process Design for Energy

**Experience**  

2011 - 2016 Energy Engineer at ENGIE UK

2017 - current Energy Data Scientist at Tempus Energy

---

### Goals for today

Introduction you to the concepts, ideas and terminology in RL

If you want to really grasp RL, you will need to dedicate significiant amount of time (same as if you want to learn NLP, convolution, GANs etc)

These notes are designed to be both
- a set of slides for lectures
- a future reference to help you learn

---

### Where to start
For those interested in learning more, any of these are a good place to start
- [Sutton & Barto - An Introduction to Reinforcement Learning (2nd Edition is in
  progress)](http://incompleteideas.net/book/bookdraft2017nov5.pdf)
- [David Silver's 10 lecture series on YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Li (2017) Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)

---
## one
### nomenclature & definitions
### background and terminology
---

### Nomenclature

[Thomas & Okal (2016) A Notation for Markov Decision Processes] (https://arxiv.org/pdf/1512.09075.pdf)  maybe put the screenshot in here

|symbol|variable  |
|------|----------|
|$s$ |state     |
|$s'$|next state|
|$a$ |action    |
|$r$ |reward    |
|$G_t$ | discounted return after time t|
|$\gamma$ |  discount factor [0, 1) |
---
### Nomenclature

|symbol|variable  |
|------|----------|
|$ a \sim \pi(s) $  | sampling action from a stochastic policy |
|$ a = \pi(s)$ | determinstic policy |
|$ \pi^\star $ | optimal policy |
|$ V_t\pi (s)$| value function |
|$ Q_t\pi (s,a)$| value function |
|$ \theta , w $ | function parameters (i.e. weights) |
|$ \mathbb{E}[f(x)] $  | expectation of f(x) |

---

### Expectation

Weighted average of all possible values - i.e. the mean

$$ \mathbb{E}[f(x)] = \sum p(x) \cdot f(x) $$

---

### Conditionals

**Probability of one thing given another**

probability of next state and reward for a given state & action

$$ P(s'|s,a) $$  

reward received from a state & action

$$ R(r|s,a,s') $$  

Sampling an action from a stochastic policy conditioned on being in state s

$$ a \sim \pi (s|a) $$


---

### Variance & bias in supervised learning
Model generalization error = <span style="color:red">bias + variance + noise</span>

**Variance**

error from sensitivity to noise in data set

model sees patterns that aren’t there -> overfitting

**Bias**

error from assumptions in the learning algorithm

model can miss relevant patterns -> underfitting

---

### Variance & bias in reinforcement learning

**Variance**

deviation from expected value

how consistent is my model / sampling

can often be dealt with by sampling more

high variance = sample inefficient

**Bias**

expected deviation vs true value

how close to the truth is my model

approximations or bootstrapping tend to introduce bias

biased away from an optimal agent / policy

---?image=assets/variance_bias.png&size=auto 90%

---
### Bootstrapping

Doing something on your own - i.e. funding a startup with your own capital

Using a function to improve / estimate itself

The Bellman Equation is bootstrapped equation

$$ V(s) = r + \gamma V(s') $$

$$ Q(s,a) = r + \gamma Q(s', a') $$

---
### Function approximation

![fig](assets/images/section_1/func_approx.png)

---
### Lookup tables
A system with two dimensions in the state variable

`state = np.array([temperature, pressure])`

|state |temperature | pressure | estimate |
|---|
|0   |high   |high   |unsafe   |
|1   |low   |high   |safe   |
|2  |high   |low   |safe   |
|3   |low   |low   |very safe   |

***Advantages***

Stability - each estimate is independent of every other estimate

***Disadvantages***

No sharing of knowledge between similar states/actions

Curse of dimensionality - high dimensional state/action spaces means lots of entries

---

### Linear functions

$$ V(s) = 3s_1 + 4s_2 $$

**Advantages**

Less parameters than a table

Can generalize across states

**Disadvantages**

The real world is often non-linear

---

###  Non-linear functions

![fig](assets/non_linear.png)

***Advantages***

Model complex dynamics

Convolution for vision

Recurrency for memory / temporal dependencies

***Disadvantages***

Instability

Difficult to train

---

### iid

Fundamental assumption in statistical learning

Independent and identically distributed

In statistical learning one always assumes the training set is independently drawn from a fixed distribution

---

### A few things about training neural networks

1 - Learning rate

2 - Batch size

3 - Scaling / preprocessing

Does anyone know what these are?

How do they affect each other?

---

### Learning rate

Controls the strength of weight updates performed by the optimizer (SGD, RMSprop, ADAM etc)

Small learning rate = slow training

High learning rate = overshoot or divergence

---

### Batch size

Modern reinforcement learning trains neural networks using batches of samples

1 epoch = 1 pass over all samples

i.e. 128 samples, batch size=64
-> two forward & backward passes across net

Smaller batch sizes = less memory on GPU

Batches train faster – weights are updated more often for each epoch

The cost of using batches is a less accurate estimate of the gradient - this noise can be useful to escape local minima

---

### Learning rate & batch sizes

Bigger batch size = larger learning rate

This is because a larger batch size gives a more accurate estimation of the gradient

https://miguel-data-sc.github.io/2017-11-05-first/

![lr_batch](assets/lr_batch.png)


---

### Scaling aka pre-processing

Neural networks don't like numbers on different scales.  Improperly scaled inputs or targets can cause issues with exploding gradients

Anything that touches a neural network needs to be within a reasonable range

Standardization = removing mean & scale by unit variance

$$ \phi(x) = x - \frac{\mu(x)}{\sigma $$}

Our data now has a mean of 0, and a variance of 1

Normalization = min/max scaling

$$ \phi(x) = \frac{x - xmin}{xmax-xmin}

Our data is now within a range of 0 to 1

---

---?image=assets/batch_norm_lit.png&size=auto 80%

### Batch normalization

In supervised learning we can estimate the true mean, variance, min or max of our data from our training set (in RL we don't get this).

Batch norm. is used in both supervised & reinforcment learning.  It's additional preprocessing of data as it moves between network layers

We use the mean and variance of the batch to normalize activations (note - standardization is actually used!)

This reduces sensitivty to weight & bias initialization

It also allows us to use higher learning rates

### Batch renormalization

Vanilla batch norm. struggles with small or non-iid batches - the mean/variance/min/max estimtates are worse

Vanilla batch norm. uses two different methods for normalization for training & testing

Batch renormalization attempts to fix this by using a single algorithm for both training & testing

---
## two
### introduction to reinforcement learning
### four central challenges
### Markov Decision Processes
---

---?image=assets/sl_unsl_rl.png&size=auto 90%

### Reinforcement learning

**one - value function methods**

Parameterize a value function - ie $V(s)$ or $Q(s,a)$

i.e. dynamic programming, SARSA, Q-Learning, DQN, DDQN

**two - policy gradient methods**

Parameterize a policy

i.e. REINFORCE, TRPO, PPO

**three - actor critic methods**

Parameterize both value functions and policies

i.e. DPG, AC2, AC3

**four - model based or planning methods**

A model can be used to plan by simulating tragetories (known as roll outs)

i.e. AlphaGo

This course covers only **model free** reinforcement learning (i.e. the first three)

---

###  Also worth knowing about are

Evolutionary methods are better able to deal with sparse error signals and eaisly parallelizable

More general optimization methods such as cross entropy method are often reccomended to be tried before you try RL

---

###  Applications

RL is all about **decision making**.

![lr_batch](assets/applications_silver.png)

[*David Silver – Deep Reinforcement Learning*](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf)

---

###  Biological inspiration

Neurobiological evidence that reward signals during perceptual learning may influence the characteristics of representations within the primate visual cortex (Mnih et. al 2015)

Habit formation

Cue -> Routine -> Reward
State -> Action -> Reward

---

###  Reinforcement learning is not

NOT an alternative method to use instead of a random forest, neural network etc

“I’ll try to solve this problem using a convolutional nn or RL” this is nonsensical

Neural networks (supervised techniques in general) are a tool that reinforcement learners can use

---

###  Deep reinforcement learning

**Deep learning** = neural networks with multiple layers

**Deep reinforcement learning** = using multiple layer networks to approximate policies or value functions

Feedforward, convolutional or recurrent neural networks are all used within different RL algorithms

---
###  A new level of intelligence

Founder & CEO of DeepMind Demis Hassabis on the brilliance of AlphaGo in it's 2015 series

![Video](assets/video/move37.mp4)

---
## Reinforcement Learning

###is

## learning through action
---

---?image=assets/images/section_2/mdp_schema_simple.png&size=auto 80%

---?image=assets/images/section_2/rl_process.png&size=auto 80%

---
### Data in reinforcement learning

In RL we generate our own data

Data = the agent's experience $(s,a,r,s')$

It's not clear what we should do with this data - no implicit target
---
### Supervised learning versus reinforcement learning

In supervised learning we are limited by our dataset

In reinforcement learning we can generate more data by acting

> Deep RL is popular because it’s the only area in ML where it’s socially acceptable to train on the test set

---
### Reinforcement learning dataset

Experience (aka a transition) $(s,a,r,s')$

$$[experience,
experience,
experience,
...
experience]$$

The dataset we generate is a sequence of experience
---
### Reinforcement learning dataset

$$[(s_0,a_0,r_1,s_1'),
(s_1,a_1,r_2,s_2'),
(s_2,a_2,r_3,s_3'),
...
(s_n,a_n,r_{n+1},s_{n+1'})]$$

What should we do with this dataset?

---
### Four central challenges in reinforcement learning

one - exploration vs exploitation

two - data quality

three - credit assignment

four - sample efficiency
---
### Exploration vs exploitation
Do I go to the restaurant in Berlin I think is best – or do  I try something new?

Exploration  = finding information
Exploitation = using information

Agent needs to balance between the two
---
### Exploration vs exploitation

How stationary are the environment state transition and reward functions?  How stochastic is my policy?

Design of reward signal vs. exploration required

Algorithm does care about the time step
 too small = rewards are delayed = credit assignment harder
 ---
### Data Quality

All the samples collected on a given episode are correlated (along the state trajectory)

This *breaks the iid assumption of independent sampling*

Environment can be non-stationary

Learning changes the data we see
Exploration changes the data we see

All of these *break the identically distributed assumption* of iid
---
## Reinforcement learning will always break supervised learning assumptions about data quality
---
###  Credit assignment

Which actions give us which reward

Reward signal is often

*delayed*	benefit of action only seen much later  

*sparse* experience with reward = 0

Sometimes we can design a more dense reward signal for a given environment
---
### Sample efficiency
How quickly a learner learns

How often we reuse data
- do we only learn once or can we learn from it again
- can we learn off-policy

How much we squeeze out of data
- i.e. learn a value function, learn a environment model
---
### Four challenges
**exploration vs exploitation**
 how good is my understanding of the range of options

**data**
 biased sampling, non-stationary distribution

**credit assignment**
which action gave me this reward

**sample efficiency**
 learning quickly, squeezing information from data
---
### Markov Decision Processes

Mathematical framework for the reinforcement learning problem

The Markov property is often a requirement to gurantee convergence
---
### Markov property

Future is conditional only on the present

Can make prediction or decisions using only the current state

Any additional information about the history of the process will not improve our decision

$$ P(s_{t+1}|s_t,a_t) = P(s_{t+1}|s_t,a_t...s_0,a_0)$$
---
### Formal definition of a MDP

$$ (\mathcal{S}, \mathcal{A}, \mathcal{R}, P, R, d_0, \gamma) $$

Set of states $\mathcal{S}$

Set of actions $\mathcal{A}$

Set of rewards $\mathcal{R}$

State transition function $ P(s'|s,a) $

Reward transition function $ R(r|s,a,s') $  

Distribution over initial states $d_0$

Discount factor $\gamma$
---
### Object oriented definition of a MDP

Two objects - the agent and Environment

Three signals - state, action & reward

```
class Agent
class Environment

state = env.reset()

action = agent.act(state)

reward, next_state = env.step(action)
```
---
### Environment

Can be real or virtual - modern RL makes heavy use of virtual environments to generate lots of experience

Each environment has a state space and an action space

Both of these spaces can be discrete or continuous

Environments can be episodic (terminating at a certain point) or continuous

The MDP framework unites both in the same way by using the idea of a final absorbing state at the end of episodes
---
### Discretiziation

Too coarse -> non-smooth control output

Too fine -> curse of dimensionality = computational expense

Discretization requires some prior knowledge

When we discretize we lose the shape of the space
---
### State

Infomation for the agent to *choose next action* and to *learn from*

State is a flexible concept - it's a n-d array
```
state = np.array([temperature, pressure])

state = np.array(pixels).reshape(height, width)
```
Possible to concactenate sequential samples together to give some idea of the recent trajectory
---
### State versus observation

I choose to distinguish between state and observation

Many problems your agent won't be able to see everything that would help it learn - i.e. non-Markov.  This then becomes a POMDP

```
state = np.array([temperature, pressure])

state = np.array([temperature, pressure])

observation = np.array([temperature + noise])
```
---
### Reward

Scalar

Delayed

Sparse

A well defined reward signal is often a limit for applications of RL (i.e. autonomous driving - whats the reward?)
---
### Agent

Our agent is the **learner and decision maker**

It's goal is to maximize total discounted reward

An agent always has a policy - even if it's a bad one
---
### The reward hypothesis

Maximising return is making an assumption about the nature of our goals

We are assuming that *goals can be described by the maximization of expected cumulative reward*

Do you agree with this?
---
### Policy $\pi(s)

A policy is rules to select actions

$pi(s)
$\pi(s,a,/theta)$
$\pi_\theta(s|a)$

Example policies
- act randomly
- always pick a specific action
- the optimal policy - the policy that maximizes future reward

Policies can be deterministic or stochastic

Policy can be
- parameterized directly (policy gradient methods)
- generated from a value function (value function methods)
---
### On versus off policy learning

![fig](assets/images/section_1/on_off_policy.png)
---
### Environment model

Our agent can optionally learn an environment model

Predicts environment response to actions
- predicts $s,r$ from $s,a$

```
def model(state, action):
# do stuff
return next_state, reward
```

Sample vs. distributional model

Model can be used to simulate trajectories for **planning**
---







MDP
















### Today - Practical

The practical we will do this afternoon is to play with a working DQN (Deep Q-Network) agent on the Open AI Cartpole
environment.

The ideas behind this practical are:
- in industry you won't be handed a set of notebooks to shift-enter through

- you will likely be given an existing code base and be expected to figure out how it works

- this skill is also useful for understanding open source projects

- using a working system allows you to understand the effect of hyperparameters

- the agent is built using TensorFlow and we will be using TensorBoard for visualizing results

---
---?image=assets/images/section2/learn_plan_act.png&size=auto 80%

*Sutton & Barto - Reinforcement Learning: An Introduction*

---?image=assets/images/section2/mdp_schema_complex.png&size=auto 80%

### Return

Goal of our agent is to maximize total future discounted reward

Return ($G_t$) is the total discounted future reward

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}

` return = reward + discount * reward + discount^2 * reward ...`

**Why do we discount future rewards?**
---
### Discounting

Future is uncertain - stochastic environment

Matches human thinking - hyperbolic discounting

Finance - time value of money

Makes return for infinite horizon problems finite

if our discount rate is $[0,1)$ then we can make the sum of an infinite series finite (known as a geometric series)

Can use discount = 1 for
- games with tree-like structures (without cycles)
- when time to solve is irrelevant (i.e. a board game)

---
## three
### introduction to value functions 
### Bellman Equation 
---
# Value function

$V_\pi(s)$

## how good is this state

# Action-value function

$Q_\pi(s,a)$

## how good is this action
---
### Value functions

Value function

$V_\pi(s) = $











---
TODO the picture of the literature

---

### Prioritized Experience Replay

Naive experience replay randomly samples batches of experience for learning.  This random sampling means we learn from experience at the same frequency as they are experienced

Some samples of experience are more useful for learning than others

We can measure how useful experience was by the temporal difference error

$$ td_error = Q(s,a) - r + \gamma Q(s', a)$$

---

### Prioritized Experience Replay

Non-random sampling introduces two problems

1 - loss of diversity - we will only sample from high TD error experiences

2 - introduce bias - non-independent sampling

Schaul et. al (2016) solves these problems by:

1 - correct the loss of diversity by making the prioritization stochastic

2 - correct the bias using importance sampling

---

### Importance Sampling

Not a sampling method - it's a method of Monte Carlo approximation

https://www.youtube.com/watch?v=S3LAOZxGcnk

Monte Carlo approximates using the sample mean, assuming that the sampling distribution (x<sub>p</sub>) is the same as
the true distribution (x~p)

$$ \mathbb{E}[f(x)] = 1/n \sum f(xi) $$

Could we use infomation about another distribution (q) to learn the distribution of p

i.e. correct for the fact that we are using another distribution

---

### Importance Sampling

The importance weight function:
$$ w(x) = p(x) / q(x) $$

$$ \mathbb{E}[f(x)] = 1/n \sum f(xi) w(xi) $$

This is an unbiased approximation, and can also be lower variance than using the sample distribution p












###  Deep Reinforcement Learning Doesn't Work Yet  

Blog post - [Deep Reinforcment Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)

State of the art reinforcement learning is **sample inefficient** - we need lots of experience to learn

Tackling any problem where we don't have access to a simulator remain beyond modern RL

**Domain specific algorithms often work faster & better**.  This is especially true if you have access to a good environment model to plan with

Requirement of a reward function - or the requirement to design one

Results can be unstable and hard to produce (this applies to a lot of scientific literature).  Different random seeds can lead to dramatically different results

Andrej Karpathy (when he was at OpenAI)
>>[Supervised learning] wants to work. Even if you screw something up you’ll usually get something non-random back. RL must be forced to work. If you screw something up or don’t tune something well enough you’re exceedingly likely to get a policy that is even worse than random. And even if it’s all well tuned you’ll get a bad policy 30% of the time, just because.

Still immature in real world production systems - examples are rare

Requirements and/or nice to haves for learning
- easy to generate experience
- simple problem
- ability to introduce self play
- well defined rewards and dense

RL solution doesn’t have to achieve a global optima, as long as its local optima is better than the human baseline
