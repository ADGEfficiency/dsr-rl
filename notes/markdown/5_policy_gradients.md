## Policy gradients

Previously we generated a policy from a value function 

$$ a = \underset{a}{argmax} Q(s,a) $$

In policy gradients we **parameterize a policy directly**

$$ a \sim \pi(a_t|s_t;\theta) $$
![Policy gradients in context](../../assets/images/section_5/intro.png){ width=30%, height=30% }

### John Schulan - Berkley, Open AI

<iframe width="560" height="315" src="https://www.youtube.com/embed/PtAIh9KSnjo?rel=0&amp;showinfo=0&amp;start=2905" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Motivations for policy gradients

### Stochastic policies

[A determinstic policy (i.e. always rock) is eaisly exploited](../..assets/images/section_5/rock_paper.png){ width=30%, height=30% }

A stochastic policy means exploration is built into the policy
- exploration can be controlled by the agent

### High dimensional action spaces

Q-Learning requires a discrete action space to argmax across

Lets imagine controlling a robot arm in three dimensions in the range [0, 90] degrees

This corresponds to approx. 750,000 actions a Q-Learner would need to argmax across

We also lose shape of the action space by discretization

![fig](../../assets/images/section_5/disc_cont_act.png){ width=30%, height=30% }

### Optimize return directly

When learning value functions our optimizer is working towards improving the predictive accuracy of the value function
- our gradients point in the direction of predicting return

This isn't what we really care about - we care about maximizing return

Policy methods optimize return directly
- changing weights according to the gradient that maximizes future reward
- aligning gradients with our objective (and hopefully a business objective)

### Simplicity

Sometimes it's eaiser to pick an action

- rather than to quantify return for each action, then pick action

Policy gradients are more general and versatile

More compatible with recurrent neural networks.  Policy gradient methods are often trained using sequeneces of experience.

## Policy gradients versus value functions

**Policy gradients**

- optimize return directly
- work in continuous and discrete action spaces
- works better in high-dimensional action spaces

**Value functions**

- optimize value function accuracy
- off policy learning
- exploration
- better sample efficiency

![fig](../../assets/images/section_5/motivation_simple.png)

## Parameterizing policies

The type of policy you parameterize depends on the **action space**

![fig](../../assets/images/section_5/discrete_policy.png){ width=30%, height=30% }

The type of policy you parameterize depends on the **action space**

![fig](../../assets/images/section_5/cont_policy.png){ width=30%, height=30% }

## Policy gradients without equations

We have a parameterized policy

- a neural network that outputs a distribution over actions

How do we improve it - how do we learn?

- change parameters to take actions that get more reward
- change parameters to favour probable actions

Reward function is not known

- but we can calculate the *gradient the expected reward*

## Policy gradients with a few equations

Our policy $\pi(a_t|s_t;\theta)$ is a **probability distribution over actions**

How do we improve it?  
- change parameters to take actions that get more reward
- change parameters to favour probable actions

Reward function is not known
- but we can calculate the *gradient of the expectation of reward*

$$\nabla\_{\theta} \mathbf{E}[G\_t] = \mathbf{E}[\nabla\_{\theta} \log \pi(a|s) \cdot G\_t]$$

We can figure out how to change our parameters without actually knowing the reward function itself

## The score function in statistics

The **score function** comes from using the log-likelihood ratio trick

The score function allows us to get the gradient of a function by **taking an expectation**

Expectataions are averages 
- use sample based methods to approximate them

$$\nabla_{\theta} \mathbf{E}[f(x)] = \mathbf{E}[\nabla_{\theta} \log P(x) \cdot f(x)]$$

### Deriving the score function

![fig](../../assets/images/section_5/score_derivation.png){ width=30%, height=30% }

**http://karpathy.github.io/2016/05/31/rl/**

## The score function in reinforcement learning

$$\nabla_{\theta} \mathbf{E}[G_t] = \mathbf{E}[\nabla_{\theta} \log \pi(a|s) \cdot G\_t]$$

` gradient of return = expectation of the gradient of the policy * return`

The RHS is an expectation - we can estimate it by sampling

The expectation is made up of things we can sample from

- we can sample from our policy 
- we can sample the return (from experience)

## Training a policy

We use the score function to get the gradient, then follow the gradient 

`gradient = log(probability of action) * return`

`gradient = log(policy) * return`

The score function limits us to on-policy learning 
- we need to calculate the log probability of the action taken by the policy

## Policy gradient intuition

$$\nabla\_{\theta} \mathbf{E}[G\_t] = \mathbf{E}[\nabla\_{\theta} \log \pi(a|s) \cdot G\_t]$$

$\log \pi(a_t|s_t;\theta)$ 
- how probable was the action we picked
- we want to reinforce actions we thought were good

$ G_t $ 
- how good was that action
- we want to reinforce actions that were actually good

Different methods to approximate the return $G_t$

We can use a Monte Carlo estimate - this is known as REINFORCE 

Using a Monte Carlo approach comes with all the problems we saw earlier
- high variance
- no online learning
- requires episodic environment

How can we get some the advantages of Temporal Difference methods?

## Baseline

We can introduce a baseline function 

- this reduces variance without introducing bias
- a natural baseline is the value function (weights $w$).  

$\log \pi(a_t|s_t;\theta) \cdot (G_t - B(s_t; w))$ 

This also gives rise to the concept of **advantage**
- how much better this action is than the average action (policy & env dependent)

$$A\_{\pi}(s\_t, a\_t) = Q\_{\pi}(s\_t, a\_t) - V\_{\pi}(s\_t)$$

### Actor-Critic

![fig](../../assets/images/section_5/ac_sum.png){ width=30%, height=30% }

Actor-Critic brings together value functions and policy gradients

We parameterize two functions
- **actor** = policy
- **critic** = value function

We update our actor (i.e. the behaviour policy) in the direction suggested by the critic

![Sutton & Barto](../../assets/images/section_5/ac_arch.png){ width=30%, height=30% }

### Actor-Critic Algorithm

![Sutton & Barto](../../assets/images/section_5/ac_algo.png){ width=30%, height=30% }

## Deterministic Policy Gradient

![fig](../../assets/images/section_5/dpg_lit.png){ width=30%, height=30% }

Actor Critic

Determinstic policy - more efficient than stochastic

Continuous action spaces

Off-policy learning

Uses experience replay

Uses target networks

### Stochastic vs determinstic policies

Stochastic policy is a probability distribution over actions

Actions are selected by sampling from this distribution

$$ \pi_{\theta}(a|s) = P[a|s;\theta] $$

$$ a \sim \pi_{\theta}(a|s) $$

DPG parameterizes a determinstic policy

$$ a = \mu_{\theta}(s) $$

### DPG components

Actor
- off policy
- function that maps state to action
- exploratory

Critic
- on-policy
- critic of the current policy
- estimates $Q(s,a)$

### Gradients

![fig](../../assets/images/section_5/DPG_grads.png){ width=30%, height=30% }

Stochastic integrates over both the state & action spaces

Deterministic integrates over only the state space -> leading to better sample efficiency

### Updating policy weights

![fig](../../assets/images/section_5/DPG_update.png){ width=30%, height=30% }

[DPG results - the difference between stochastic (green) and deterministic (red) increases with the dimensionality of the action space](../../assets/images/section_5/DPG_results.png){ width=30%, height=30% }

## A3C

![fig](../../assets/images/section_5/A3C_lit.png){ width=30%, height=30% }

Asynchronous Advantage Actor-Critic 

We saw earlier that experience replay is used to make learning more stable & decorrelate updates
- but can only be used with off-policy learners

**Asynchronous**

- multiple agents learning separately
- experience of each agent is independent of other agents
- learning in parallel stabilizes training
- allows use of on-policy learners
- runs on single multi-core CPU
- learns faster than many GPU methods

**Advantage** 

- the advantage function 

$$A\_{\pi}(s\_t, a\_t) = Q\_{\pi}(s\_t, a\_t) - V\_{\pi}(s\_t)$$

How much better an action is than the average action followed by the policy

### A3C algorithm

![fig](../../assets/images/section_5/A3C_algo.png){ width=30%, height=30% }

https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
