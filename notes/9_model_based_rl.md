# These notes are under development!

https://www.youtube.com/watch?v=8a7wBLg5Q8U - microsoft keynote model based rl

[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://sites.google.com/view/drl-in-a-handful-of-trials)

Jonathan Hui - Model-based Reinforcement Learning - [blog post](https://medium.com/@jonathan_hui/rl-model-based-reinforcement-learning-3c2b6f0aa323)

# Temporal difference model (Pong 2018) notes

Convential wisdom is that model free is less efficient but with the best asymptotic performance, while model based are more efficient but less optimal policies.  This is due to bias in the dynamics model hurting the policy.

# Investigating Model Based RL for Continuous Control | Alex Botev - [youtube](https://www.youtube.com/watch?v=1_sYif82CtY)

Model based RL = we try to learn an environment dynamics model.  Usually assume we don't have access to the environment except through sampling.

Why model based rl

- can solve any task without interaction
- task independent - model can be used for different tasks (model free is reward dependent)
- usually trained using supervised (sometimes unsupervised) techniques, which are stable
- can use sophisticated algorithms such as trajectory optimization or tree search
- better use of gathered data (more sample efficient)

Learning models is hard - currently model free is more sample efficient, with better asymptotic behaviour.

A key challenge is short versus long term consistency.  One step predictions can be very accurate, but unrolling the environment model independent of the environment can be very difficult.

Why dynamics models are hard to train

- not all aspects of the environment are relevant (i.e. a part of the screen that isn't relevant to the task)
- compounding errors lead to inaccurate long term predictions - this is one of the main issues at the moment (if your task requires long horizons to plan over)
- hard to estimate uncertainty for flexible models
- neural networks need lots of data to learn - this makes model based reinforcement learning less sample efficient

Worked on *value expansion for actor-critic* - using a learned dynamics model to create on-policy targets & multi-step horizon targets.  Idea is that more targets can be more stable.

Takeaways

- ensembles of dynamics models was necessary
- training on model step losses was necessary
- multi-step expansion was necessary
- being pessimistic with respect to the value expansion targets can improve performance (i.e taking a minimum over horizons, rather than average) - this is similar to DDQN where you are trying to stop overestimation bias

Stochastic models require single step losses - deterministic models can be trained with multiple step losses.

# Dyna 2 Notes

1. learning = interacting with env
2. improving policy without env interaction

Monte Carlo = random policy, argmax across experienced returns

## MCTS

1. greedy action selection within the tree
2. random action selection until termination

If simulated state is full expanded then agent selects greedy action

After each simulation, action values are updated to the average MC return.  In practice, only one new state-action pair is added per episode

## UCT

Improves the greedy seelction.  Each state is treated as a multi-armed bandit, actions chosen using UCB to balance exploration + exploitation

# Sutton Barto

## Chapter 8 - Planning + learning with tabular methods

Sample vs distributional env models

- dist = all possible results + their probabilities (used in dynamic programming)
- dist = stronger (you can sample from them)
- eaiser to obtain sample models

Two types of planning

1. state space planning (used in this book)
2. plan space planning

The basic structure of state space planning

1. computing value functions to improve the policy
2. compute value functions by updates or backups applied on simulated experience

Dynamic programming fits this structure

This common structure ties together model free and model based.  Differences in how performance is assessed and how flexibly experience can be generated

One step tabular Q-planning

```python
while not done:
    state = env.state_space.sample()
    action = env.action_space.sample()

    next_state, reward = model.step(state, action)

    q_table.update(state, action, reward, next_state)
```

Another theme is **benefits of planning in small, incremental steps**.  This means planning can be interrupted/redirected with little wasted computation

### 8.2 - Dyna

Two roles for real experience

1. direct rl
2. model learning (indirect)

Direct = simpler, not affected by model bias
Indirect = better policy with fewer samples

![Relationships amoung learning, planning and acting (Fig 8.1 - Sutton & Barto)](../../assets/images/section_9/fig1.png){ width=60%, height=60% }

In Dyna

- planning = random sample one step tabular Q-Learning
- direct = one step tabular Q-Learning
- model learning = table based (assumes deterministic env)

Model returns last observed next state and reward as it's prediction

During planning, Q-Planning randomly samples from state-action pairs that have been experienced

![The general Dyna Architecture. Real experience, passing back and forth between the environment and the policy, affects policy and value functions in much the same way as does simulated experience generated by the model of the environment.](../../assets/images/section_9/fig1.png){ width=60%, height=60% }

Only difference between learning and planning is the source of the experience

### Tabular Dyna-Q

Learning and planning accomplished by the same algorithm.  

```python
state = env.reset()
while not done:
    action = e_greedy(state, Q)
    r, next_state = env.step(action)
    Q.update(state, action, reward, next_state)  # max update
    model.update(state, action, reward, next_state)

    for n in range(n):
        state = model.sample_state()
        action = model.sample_action(state)
        reward, next_state = model.predict(state, action)
        Q.update(state, action, reward, next_state)
```

### When the model is wrong

Model can be incorrect because

- environment is stochastic
- limited number of samples
- model fitted improperly
- non-stationary

Sometimes model error will be evident (if model is optimisitic) othertimes the model error won't be found (policy that misses the chance to exploit new oppourtunities)

DynaQ includes a bonus reward on simulated experience for untested actions - adds a small bonus `r + k \sqrt{\tau\}`

### Prioritized sweeping

Planning can be more efficient if simulated transitions are focused on particular state pairs

Prioritized sweeping = a queue of state-action pairs prioritized by the size of the change

In stochastic environments the update is an expected (not sample) update.

Sampling updates can win because they break up the backup into smaller pieces - corresponding to individual transitions.  

### Expected vs. sample updates

Three dimensions for updates
1. state or action value
2. on or off policy
3. expected or sample

Expected updates are only possible with a distribution model.  Expected updates require more computation (which is often the limiting resource in planning)

Difference between expected and sample depends on how stochastic the environment is.  Sample update has sampling error, but cheaper.  

Expected is roughly b times more expensive - where b is the branching factor (b = number of possible next states where probability != 0)

Sample updates work faster + the fact that values of successor states are updated = sample updates likely to be superior

### Trajectory sampling

Updating state-action pairs according to a distribution.  One possibility is to update according to the on-policy distribution - the distribution of state-actions observed by the current policy.

Allows ignoring the state space

# [Lecture 9 - Model-based Reinforcement Learning - Chelsea Finn](https://www.youtube.com/watch?v=iC2a7M9voYU)

### Sample efficiency

Methods

- gradient free (NES, CMA)
- fully online (A3C) 
- policy gradients
- replay buffer value functions
- model based deep rl (guided policy search)
- model based shallow rl (PILCO)

Each one is 10x more sample efficient

Grad free 10x slower than fully online
Fully online 10x slower than policy grads

PG -> DQN = 10x faster

Model based = 10x faster

### Transferability and generality

Learning model that can be used for different tasks.  A policy can be used for only one task

Use supervised learning to fit the model to observations from the environmentc

### Backproping gradients through model

Backprop expected reward through model into policy

1. run random policy to collect transitions
2. learn model to minimize mean square error
3. backprop derivatives into the policy

Base policy is important

Effective if you can hand engineer dynamics using physics and fit a few parameters

In many cases - doesn't work

TODO

### Local models + guided policy search
