### Practical <a id="section-practical"></a>

Experiments with a working DQN agent & the Open AI CartPole environment

---
### Practical <a id="section-practical"></a>

- you won't be handed a set of notebooks to shift-enter through

- given an existing code base and be expected to figure out how it works

- learn to read other peoples code in the wild

- useful for understanding open source projects

- using a working system allows you to understand the effect of hyperparameters and feel how hard RL can be!

---
###  CartPole

<iframe width="560" height="315"
src="https://www.youtube.com/embed/46wjA6dqxOM?rel=0&showinfo=0&autoplay=1&loop=1&playlist=46wjA6dqxOM" frameborder="0"
allow="autoplay; encrypted-media" allowfullscreen></iframe>

CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

---
### Hyperparameters

Hyperparameters are configured using a dictionary
- using dictionaries to setup agents/experiments allows you to eaisly save them to a text file
- you can also explode a dictionary into a function

```
config = {'param1': 10, 'param2': 12}

def expt(param1, param2):
    return param1 * param2

>>> expt(**config)
120
```

---
### Hyperparameters

What do you think the effect of changing each of these hyperparameters will be?

```
config_dict = {'discount': 0.97,
               'tau': 0.001,
               'total_steps': 500000,
               'batch_size': 32,
               'layers': (50, 50),
               'learning_rate': 0.0001,
               'epsilon_decay_fraction': 0.3,
               'memory_fraction': 0.4,
               'process_observation': False,
               'process_target': False}
```



