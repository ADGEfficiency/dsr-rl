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

2006 - 2011 B.Eng Chemical Engineering, MSc Advanced Process Design for Energy

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
|s     | state      |
|s'    | next state |

|$$s$$ |state     |
|$$s'$$|next state|

$$a$$ action

$$r$$ reward

$$G_t$$ discounted return after time t

$$gamma$$ discount factor [0, 1)

$$a ~ pi(s)$$
