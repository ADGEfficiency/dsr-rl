### Recap

Three sources of generalization error
- ?
- ?
- ?

Missing relevant patterns in data = ?

Seeing patterns that aren't there = ? 

One advantage & disadvantage of lookup tables
- advantage = ?
- disadvantage = ? 

iid = ? and ? distributed 

Larger batches -> ? learning rate

Why do we pass in `None` for the first dimension in TensorFlow
`tf.placeholder(shape=(None, 14, 2))`

### Recap answers

Three sources of generalization error
- bias
- variance
- noise

Missing relevant patterns in data = bias

Seeing patterns that aren't there = variance 

One advantage & disadvantage of lookup tables
- advantage = stability
- disadvantage = no aliasing between states, curse of dimensionality

iid = independent and identically distributed 

Larger batches -> larger learning rate
- better estimation of the gradient

Why do we pass in `None` for the first dimension in TensorFlow
`tf.placeholder(shape=(None, 14, 2))`

- first dimension is the batch dimension
