import numpy as np

from qfunc import Qfunc

config = {'input_shape': (2, 2),
          'output_shape': (2,),
          'layers': [(3,3), (3,3)]}

input_shape = config['input_shape']
output_shape = config['output_shape']
layers = config['layers']

#  this line of code will go into the Qfunc class
# input_shape = (None,) + layers[0]
input_shape = layers[0]

input = np.ones(input_shape)
weights = np.ones(layers[0])
bias = np.zeros(layers[0])

pre_act = np.dot(input, weights) + bias

relu = pre_act * (pre_act > 0)

