## nomenclature & statistics background

### Nomenclature

Nomenclature in RL can be inconsistent
- value function methods, action = `a`
- policy gradient methods, action = `u`

Following [Thomas & Okal (2016) A Notation for Markov Decision Processes](https://arxiv.org/pdf/1512.09075.pdf)


|symbol | variable  |
|$s$ |state     |
|$s'$|next state|
|$a$ |action    |
|$r$ |reward    |
|$G_t$ | discounted return after time t|
|$\gamma$ |  discount factor [0, 1) |
|$ a \sim \pi(s) $  | sampling action from a stochastic policy |
|$ a = \pi(s)$ | deterministic policy |
|$ \pi^\star $ | optimal policy |
|$ V_\{\pi} (s)$| value function |
|$ Q_\{\pi} (s,a)$| value function |
|$ \theta , \omega $ | function parameters (weights) |
|$ \mathbf{E}[f(x)] $  | expectation of f(x) |

### Expectations

Weighted average of all possible values (the mean)

`expected_value = probability * magnitude`

$$ \mathbf{E} [f(x)] = \sum p(x) \cdot f(x) $$

Expectations **allow us to approximate by sampling**

- if we want to approximate the average time it takes us to get to work 

- we can measure how long it takes us for a week and get an approximation by averaging each of those days

### Conditionals

**Probability of one thing given another**

probability of next state and reward given state & action

$$ P(s'|s,a) $$  

reward received from a state & action

$$ R(r|s,a,s') $$  

sampling an action from a stochastic policy conditioned on being in state $s$

$$ a \sim \pi (s|a) $$

### Variance & bias in supervised learning

Model generalization error = <span style="color:red">bias + variance + noise</span>

**Variance**

- error from sensitivity to noise in data set
- seeing patterns that aren’t there -> overfitting

**Bias**

- error from assumptions in the learning algorithm
- missing relevant patterns -> underfitting

### Variance & bias in RL 

**Variance** = deviation from expected value

- how consistent is my model / sampling
- can often be dealt with by sampling more
- high variance = sample inefficient

**Bias** = expected deviation vs true value

- how close to the truth is my model
- approximations or bootstrapping tend to introduce bias
- biased away from an optimal agent / policy

![fig](assets/images/section_1/variance_bias.png)

### Bootstrapping

Doing something on your own 
- i.e. funding a startup with your own capital
- using a function to improve / estimate itself

The Bellman Equation is bootstrapped equation

$$ V(s) = r + \gamma V(s') $$

$$ Q(s,a) = r + \gamma Q(s', a') $$

Bootstrapping often introduces bias 
- the agent has a chance to fool itself 

### Function approximation

![fig](assets/images/section_1/func_approx.png)

### Lookup tables
Two dimensions in the state variable

`state = np.array([temperature, pressure])`

|state |temperature | pressure | estimate |
|0   |high   |high   |unsafe   |
|1   |low   |high   |safe   |
|2  |high   |low   |safe   |
|3   |low   |low   |very safe   |

### Lookup tables

**Advantages**

Stability

Each estimate is independent of every other estimate

**Disadvantages**

No sharing of knowledge between similar states/actions

Curse of dimensionality 

High dimensional state/action spaces means lots of entries

### Linear functions

$$ V(s) = 3s_1 + 4s_2 $$

**Advantages**

Less parameters than a table

Can generalize across states

**Disadvantages**

The real world is often non-linear


###  Non-linear functions

Most commonly neural networks

**Advantages**

Model complex dynamics

Convolution for vision

Recurrency for memory / temporal dependencies

**Disadvantages**

Instability

Difficult to train

### iid

Fundamental assumption in statistical learning

**Independent and identically distributed**

In statistical learning one always assumes the training set is independently drawn from a fixed distribution

