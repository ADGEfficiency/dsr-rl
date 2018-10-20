# One - Statistical Background

Variance and bias, IID, function approximation.

---

## Nomenclature

Nomenclature in RL can be inconsistent.  Modern literature has largely settled on nomenclature as given below.  Historically some policy gradient methods would use `u` for action - this is also common in *optimal control* (a related field - see the notes for Section 2).

These notes follow [Thomas & Okal (2016) A Notation for Markov Decision Processes](https://arxiv.org/pdf/1512.09075.pdf)

|symbol | variable  |
|---| ---|
|$s$ |state     |
|$s'$|next state|
|$a$ |action    |
|$r$ |reward    |
|$G_t$ | discounted return after time t|
|$\gamma$ |  discount factor [0, 1) |
|$a \sim \pi(s)$  | sampling action from a stochastic policy |
|$a = \pi(s)$ | deterministic policy |
|$\pi^{\star}$ | optimal policy |
|$V\ {\pi} (s)$| value function |
|$Q\ {\pi} (s,a)$| value function |
|$\theta, \omega$ | function parameters (weights) |
|$\mathbb{E}[f(x)]$  | expectation of f(x) |
|$X$ | a random variable |
|$f(x;\theta)$| density of $X$ at point $x$, parameterized by $\theta$ |
|$f(x,y)$ | joint density of $X$ and $Y$ at the point $(x,y)$ |
|$f(x|y)$ | conditional distribution of $X$ given $Y$ |

## Expectations 

**An expectation is simply the mean** (or average for the less statistically careful).

` expectation = probability * magnitude `

$$ \mathbf{E} [f(x)] = \sum_{x} p(x) \cdot f(x) $$

Expectations allow us to **approximate by sampling**

Approximating the expectation for our commute to work can be done by sampling the time across three days and averaging.

Modern reinforcement learning optimizes expectations - expected future reward.  RL is trying to take actions that **on average** are the best.

## Conditionals 

**Probability of one thing given another**

Probability of next state and reward given state & action $P(s'|s,a)$

Reward received from a state & action $R(r|s,a,s')$

Sampling an action from a stochastic policy $a \sim \pi (s|a)$

Note the difference in notation for paramertizations, joint densities and conditional distributions.

|symbol | variable  |
|---| ---|
|$f(x;\theta)$| density of $X$ at point $x$, parameterized by $\theta$ |
|$f(x,y)$ | joint density of $X$ and $Y$ at the point $(x,y) $ |
|$f(x|y)$ | conditional distribution of $X$ given $Y$ |

\newpage

## Variance & bias

Model generalization error = <span style="color:red">bias + variance + noise</span>

### In supervised learning

Variance = overfitting

- error from sensitivity to noise in data set
- seeing patterns that aren’t there 

Bias = underfitting

- error from assumptions in the learning algorithm
- missing relevant patterns 

### In reinforcement learning

Variance = deviation from expected value

- how consistent is my model / sampling
- can often be dealt with by sampling more
- high variance = sample inefficient

Bias = expected deviation vs true value

- how close to the truth is my model
- approximations or bootstrapping tend to introduce bias
- biased away from an optimal agent / policy

![Variance = consistency, bias = error versus the truth](../../assets/images/section_1/variance_bias.png){ width=30%, height=30% }

\newpage

## Bootstrapping

Doing something on your own 

- funding a startup with your own capital
- using a function to improve or approximate itself

The Bellman Equation is bootstrapped equation

$$ V(s) = r + \gamma V(s') $$
$$ Q(s,a) = r + \gamma Q(s', a') $$

Bootstrapping often introduces bias.  The bootstrapped approximation gives the agent a chance to mislead itself.

## IID - independent and identically distributed

Fundamental assumption made in statistical learning

Assuming the training set is independently drawn from a fixed distribution

In the context of image classification

- independent sampling = we have photos from a wide range of sources, and each photo is independent of our other photos

- fixed distribution = the photos we have in our training set are the same kinds of photos as we will do prediction for

## Curse of dimensionality

Refers to phenomena that occur in high dimensional spaces.  High dimensional spaces means we need more data to support these dimensions.  This is known as the **combinatorial explosion**

Often we need to consider every possible combination of actions. A high dimensional action space (say a robot with multiple arms) means we need to consider a large number of potential actions.  Each additional dimension doubles the effort to consider all of the combinations.

Rule of thumb - 5 training examples for each dimension in the representation (for supervised learning).

\newpage

## Importance sampling 

[Wikipedia](https://en.wikipedia.org/wiki/Importance_sampling) - [Introduction video on Youtube](https://www.youtube.com/watch?v=S3LAOZxGcnk) - 

*Reinforcement learning context - importance sampling is used in prioritized experience replay*

Importance sampling = a variant of Monte Carlo approximation - it's a method of approximating expectations.

If we had access to a probability distribution we can approximate the expectation analytically.  We sum across all the possible values of the function $f(x)$ and multiply by the probability of that value occurring:

$$ \mathbf{E}[f(x)] = \sum_{x} p(x) \cdot f(x) $$

If we don't have access to the distribution we could instead approximate the expectation using Monte Carlo, by looking at the sample mean across $n$ samples.  Here are sampling from the true distribution $x \sim p$.  This means our sample expectation will converge to the true exepcation of the distribution $p(x)$:

$$ \mathbf{E}[f(x)] = \frac{1}{n} \sum_{i=1}^{n} f(x_i) $$

Now lets imagine that we could only take samples from a different distribution $x \sim q$.  This will have a different expectation (because the probabilities of drawing a sample $x$ are different):

$$ \mathbf{E}[f(x)] = \sum_{x} q(x) \cdot f(x) $$

The magic of importance sampling is that we can use these samples $q(x)$ to improve our approximation of the distribution we can't sample from $p(x)$.  If we multiply our true distribution by the probabilities of our sample distribution:

$$ \mathbf{E}[f(x)] = \sum_{x} p(x) \cdot f(x) \cdot \frac{q(x)}{q(x)} $$

We end up being able to take a sample expectation according to the sampling distribution $q(x)$:

$$ \mathbf{E}[f(x)] = \frac{1}{n} \sum_{i=1}^{n} f(x_i) \frac{p(x_i)}{q(x_i)} $$

The ratio of the two probabilities is called the importance weight:

$$ w(x) = \frac{p(x)}{q(x)} $$

$$ \mathbf{E}[f(x)] = \frac{1}{n} \sum_{i=1}^{n} \cdot f(x_i) \cdot w(x_i) $$

Note that we do need to know the probabilities for both $p(x_i)$ and $q(x_i)$ - i.e. how likely the samples were.  But we only need to know $f(x)$ as sampled from the distribution $q(x)$ - i.e. we don't need to know what the function was under the true distribution $p(x)$.

## Entropy 

[Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory)) - [Must know Information Theory concepts in Deep Learning (blog post)](https://towardsdatascience.com/must-know-information-theory-concepts-in-deep-learning-ai-e54a5da9769d)

*Context - entropy used for decision tree construction*

Like it's thermodynamic counterpart, entropy has a number of interpretations

- measure of uncertainty - more uncertainty is higher entropy
- measure of predictability - less predictable (more random) is higher entropy
- average infomation gain from observing an experiment - this is a function of the probability of an outcome.  More infomation gained from rarer observations

Deterministic experiment (completely predictable, i.e. a coin with one side) has zero entropy.

Entropy is a measurement of how much information is contained in a distribution.  Entropy gives us the theoretical lower bound on the number of bits we would need to encode our infomation.  Knowing this lower bound allows us to quantify how much infomation is in our data.

Taking $log_{2}$ means we can interpret the entropy as measured in bits (i.e. 0 or 1)

$$H(x)=-\sum{i=1}{N} p(x_i) \cdot \log_{2} p(x_i)$$

Some policy gradient based agents will have an entropy maximization term in the loss function - to make the policy as random as possible.

## Cross-entropy 

[Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy) - [Must know Information Theory concepts in Deep Learning (blog post)](https://towardsdatascience.com/must-know-information-theory-concepts-in-deep-learning-ai-e54a5da9769d)

*Context - cross entropy is used as a loss function in classification neural networks - minimizing the distance between the output softmax and the true classes*

- measures the distance between or similarity of two probability distributions

For the discrete case the cross entropy is given by

$$H(p,q) = - \sum{x} \cdot log q(x)$$

## Mutual infomation 

[Wikipedia](https://en.wikipedia.org/wiki/Mutual_information) - [Must know Information Theory concepts in Deep Learning (blog post)](https://towardsdatascience.com/must-know-information-theory-concepts-in-deep-learning-ai-e54a5da9769d)

- dependency between two distributions
- how much infomation about one variable is carried by another
- measures how much knowing one of these variables reduces uncertainty about the other
- Quantifies the bits (ie amount of infomation) obtained about one variable through the other

More generalized version of the linear correlation coefficient.  Mutual dependence = 0 guarantees that random variables are independent, zero correlation doesn't.

$$I(X;Y) = \sum{y \in Y} \sum{x \in X} p(x,y) log(\frac{p(x,y)}{p(x)p(y)})$$

Note that for continuous variables, the sums are replaced by integrals.  If $log_{2}$ is used, the units are bits.

- for independent variables, the mutual infomation is zero.
- for deterministic functions, all infomation is shared -> mutual infomation is the entropy or Y (or X)

## Kullback–Leibler divergence 

[Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) - [Must know Information Theory concepts in Deep Learning (blog post)](https://towardsdatascience.com/must-know-information-theory-concepts-in-deep-learning-ai-e54a5da9769d)

*Reinforcement learning context - L divergence is used in [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf) and [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) to constrain how much a policy changes during learning.  Also used in [C51 - A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf) uses the KL divergence, and suggests that the Wasserstein metric might be a fruitful next step.*

- measures the difference between probability distributions 
- KL divergence between $P$ and $Q$ tells us how much infomation we lose when we try to approximate data given by $P$ with $Q$

$$D_{KL}(P||Q) = \mathbb{E}_{x} \cdot \log \frac{P(x)}{Q(x)}$$

Often used in reinforcement learning to measure/constrain/penalize the distance between policy updates.  Also known as relative entropy or information gain. 

\newpage

## Function approximation

![Three commonly use function approximation methods](../../assets/images/section_1/func_approx.png){ width=30%, height=30% }

### Lookup tables
Imagine a problem where we have two dimensions in the state variable, with each state variable having two discrete options (either high or low).  We use this state variable to predict the safety of the system.

`state = np.array([temperature, pressure])`

|state |temperature | pressure | safety estimate |
|---|---|---|---|
|0   |high   |high   |unsafe   |
|1   |low   |high   |safe   |
|2  |high   |low   |safe   |
|3   |low   |low   |very safe   |

Advantages

- Stability
- Each estimate is independent of every other estimate

Disadvantages

- No sharing of knowledge between similar states/actions
- Curse of dimensionality - high dimensional state and action spaces means large tables

### Linear functions

$$ V(s) = 3s_1 + 4s_2 $$

Advantages

- Less parameters than a table
- Can generalize across states

Disadvantages

- The real world is often non-linear

###  Non-linear functions

Most commonly neural networks

Advantages

- Model complex dynamics
- Convolution for vision
- Recurrency for memory / temporal dependencies

Disadvantages

- Instability
- Difficult to train

