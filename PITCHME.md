## a glance at reinforcement learning

Adam Green
[adam.green@adgefficiency.com](adam.green@adgefficiency.com)
[adgefficiency.com](http://adgefficiency.com)

---

### Course Materials
All course materials are in the GitHub repo DSR_RL.

The materials are
- lecture notes hosted on GitPages at
- a collection of useful literature at
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

### Today

RL is a vast topic, worthy of a lifetime of study!  

Today we are aiming to introduce you to the concepts, ideas and terminology in RL

If you want to really grasp RL, you will need to study it on your own!

These notes are designed as a future reference, to be looked back over when you dive deeper into the topic.

---

### Where to start
For those interested in learning more, any of these are a good place to start
- [Sutton & Barto - An Introduction to Reinforcement Learning (2nd Edition is in
  progress)](http://incompleteideas.net/book/bookdraft2017nov5.pdf)
- [David Silver's 10 lecture series on YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Li (2017) Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)

---

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

## one
### nomenclature & definitions
### background and terminology

---

### Nomenclature
|symbol|variable  |
|------|----------|
|$$s$$ |state     |
|$$s'$$|next state|
|$$a$$ |action    |
|$$r$$ |reward    |
|$$G_t$$ | discounted return after time t|
|$$\gamma$$ |  discount factor [0, 1) |
---
### Nomenclature

|symbol|variable  |
|------|----------|
|$$ a \sim \pi(s)$$ | sampling action from a stochastic policy |
|$$ a = \pi(s)$$ | determinstic policy |
|$$ \pi^\star $$ | optimal policy |
|$$ V_t\pi (s)$$| value function |
|$$ Q_t\pi (s,a)$$| value function |
|$$ \theta , w $$ | function parameters (i.e. weights) |
|$$ \mathbb{E}[f(x)] $$  | expectation of f(x) |

---

### Expectation

Weighted average of all possible values - i.e. the mean

$$ \mathbb{E}[f(x)] = \sum p(x) \cdot f(x) $$

---

### Conditionals

**Probability of one thing given another**

probability of next state and reward for a given state & action

$$ P(s',r|s,a) $$  

reward received from a state & action

$$ R(r|s,a) $$  

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

---?image=assets/func_approx.png&size=auto 90%

---
### Lookup tables

---?image=assets/lookup_table.png&size=auto 90%

---
### Lookup tables

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

---?image=assets/non_linear.png&size=auto

---

###  Non-linear functions

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

independent and identically distributed

In statistical learning one always assumes the training set is independently drawn from a fixed distribution

---

### Batch size

Modern reinforcement learning trains neural networks using batches of samples

1 epoch = 1 pass over all samples

i.e. 128 samples, batch size=64
-> two forward & backward passes across net

---

###  Batch size

Smaller batch sizes = less memory on GPU

Batches train faster – weights are updated more often for each epoch

The cost of using batches is a less accurate estimate of the gradient - this noise can be useful to escape local minima

---?image=assets/batch_norm_lit.png&size=auto 80%
