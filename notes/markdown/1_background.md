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

## <span style="color:#66ff66">a few things about training neural networks</span>

### A few things about training neural networks

Learning rate

Batch size

Scaling / preprocessing

Larger batch size 
- larger learning rate
- decrease in generalization
- increase in batch normalization performance

### Learning rate

Controls the strength of weight updates performed by the optimizer (SGD, RMSprop, ADAM etc)

$$ \theta^{t+1} = \theta^{t} - \alpha \frac{\partial E(x, \theta^{t})}{\partial \theta} $$

where $E(x, \theta^{t})$ is the error backpropagated from sample $x$

Small learning rate 
- slow training

High learning rate 
- overshoot or divergence

### Learning rate

Always intentionally set it

`from keras.models import Sequential`

`#  don't do this!`

`model.compile(optimizer='rmsprop', loss='mse')`

`#  do this`

`from keras.optimizers import RMSprop`

`opt = RMSprop(lr=0.001)`

`model.compile(optimizer=opt, loss='mse')`

### Batch size

Modern reinforcement learning trains neural networks using batches of samples

Below we have a dataset with four samples, of shape (14, 2)

`>>> import numpy as np`

`>>> data = np.arange(4*28).reshape(4, -1, 2)`

`>>> data.shape`

`(4, 14, 2)`

The first dimension is the batch dimension - this is foundational in TensorFlow 

`tf.placeholder(shape=(None, 14, 2))`

Passing in `None` allows us to use whatever batch size we want 

### Batch size

Smaller batches can fit onto smaller GPUs
- if a large sample dimension we can use less samples per batch

Batches allow us to learn faster
- weights are updated more often during each epoch

Batches give a less accurate estimate of the gradient 
- this noise can be useful to escape local minima

Larger batch size -> larger learning rate
- more accurate estimation of the gradient (better distribution across batch)
- we can take larger steps

![lr_batch](assets/images/section_1/lr_batch.png)

Larger batch size -> larger optimal learning rate

*https://miguel-data-sc.github.io/2017-11-05-first/*

### Batch size

Observed that larger batch sizes decrease generalization performance 

Poor generalization  due to large batches converging to *sharp minimizers* 
- areas with large positive eigenvalues $ \nabla^{2} f(x) $
- Hessian matrix (matrix of second derivatives) where all eigenvalues positive = positive definite = local minima

Batch size is a **hyperparameter that should be tuned**

*https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network*

### Scaling aka pre-processing

Neural networks don't like numbers on different scales  
- improperly scaled inputs or outputs can cause issues with gradients
- anything that touches a neural network needs to be within a reasonable range

We can estimate statistics like min/max/mean from the training set
- these statistics are as much a part of the ML model as weights
- in reinforcement learning we have no training set

### Scaling aka pre-processing

**Standardization** = removing mean & scale by unit variance

$$ \phi(x) = x - \frac{\mu(x)}{\sigma(x)} $$

Our data now has mean of 0, variance of 1

**Normalization** = min/max scaling

$$ \phi(x) = \frac{x - x\_{min}}{x\_{max} - x\_{min}} $$

Our data is now between 0 and 1

![fig](assets/images/section_1/batch_norm_lit.png)

### Batch normalization

Batch norm. is additional preprocessing of data as it moves between network layers
- used in very deep convolutional/residual nets

We use the mean and variance of the batch to normalize activations 
- standardization is actually used!
- reduces sensitivty to weight & bias initialization
- allows higher learning rates
- originally applied before the activation - but this is a topic of debate

[Batch normalization before or after relu - Reddit](http://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)

[Ian Goodfellow Lecture (3:20 onward)](http://www.youtube.com/watch?time_continue=385&v=Xogn6veSyxAppl)

### Batch renormalization

Vanilla batch norm. struggles with small or non-iid batches 
- the estimated statistics are worse

- vanilla batch norm. uses two different methods for normalization during training & testing

- batch renormalization uses a single algorithm for both training & testing
