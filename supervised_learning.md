Feature vs target



Regression vs classification

Super vs unsuper vs rl


---

ml is non-stationary - what is true today may not be true tomorrow
- amount of data required

---

## one
### nomenclature & statistics background
### a few things about training neural networks

---
### A few things about training neural networks

Learning rate

Batch size

Scaling / preprocessing

Larger batch size
- larger learning rate
- decrease in generalization
- increase in batch normalization performance

---
### Learning rate

Controls the strength of weight updates performed by the optimizer (SGD, RMSprop, ADAM etc)

$$ \theta^{t+1} = \theta^{t} - \alpha \frac{\partial E(x, \theta^{t})}{\partial \theta} $$

where $E(x, \theta^{t})$ is the error backpropagated from sample $x$

Small learning rate
- slow training

High learning rate
- overshoot or divergence

---
### Learning rate

Always intentionally set it

```python
from keras.models import Sequential

#  don't do this!
model.compile(optimizer='rmsprop', loss='mse')

#  do this
from keras.optimizers import RMSprop
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt, loss='mse')
```

---
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

---
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

---

<img src="assets/images/section_1/lr_batch.png" height="80%" width="80%" align="top">

<div class="image_footer">https://miguel-data-sc.github.io/2017-11-05-first/</div>

---
### Batch size

Observed that larger batch sizes decrease generalization performance

Poor generalization  due to large batches converging to *sharp minimizers*

- areas with large positive eigenvalues $ \nabla^{2} f(x) $
- Hessian matrix (matrix of second derivatives) where all eigenvalues positive = positive definite = local minima

Batch size is a **hyperparameter that should be tuned**

---
### Scaling aka pre-processing

Neural networks don't like numbers on different scales
- improperly scaled inputs or outputs can cause issues with gradients
- anything that touches a neural network needs to be within a reasonable range

We can estimate statistics like min/max/mean from the training set
- these statistics are as much a part of the ML model as weights
- in reinforcement learning we have no training set

---
### Scaling aka pre-processing

**Standardization** = removing mean & scale by unit variance

$$ \phi(x) = x - \frac{\mu(x)}{\sigma(x)} $$

Our data now has mean of 0, variance of 1

**Normalization** = min/max scaling

$$ \phi(x) = \frac{x - x\_{min}}{x\_{max} - x\_{min}} $$

Our data is now between 0 and 1

---
![fig](assets/images/section_1/batch_norm_lit.png)

---
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

---
### Batch renormalization

Vanilla batch norm. struggles with small or non-iid batches

- the estimated statistics are worse
- vanilla batch norm. uses two different methods for normalization during training & testing
- batch renormalization uses a single algorithm for both training & testing
