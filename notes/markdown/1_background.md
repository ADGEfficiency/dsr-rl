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
|$a \sim \pi(s)$  | sampling action from a stochastic policy |
|$a = \pi(s)$ | deterministic policy |
|$\pi_^{\star}$ | optimal policy |
|$V_\{\pi} (s)$| value function |
|$Q_\{\pi} (s,a)$| value function |
|$\theta, \omega$ | function parameters (weights) |
|$\mathbf{E}[f(x)]$  | expectation of f(x) |
