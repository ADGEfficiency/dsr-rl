import numpy as np
import tensorflow as tf
from qfunc import Qfunc

config = {'input_shape': (2,),
          'layers': (4,2),
          'output_shape':(2,)}

input_shape = config['input_shape']
layers = config['layers']
output_shape = config['output_shape']

def weight(shape): return np.ones(shape)
def bias(shape): return np.ones(shape)
def relu(x): return x * (x > 0)

def test_network():
    input = np.random.uniform(size=input_shape)
    win = weight((*input_shape, layers[0]))
    bin = bias(layers[0])
    preact_in = np.matmul(input, win) + bin
    input_layer = relu(preact_in)

    w1 = weight((layers[0], layers[1]))
    b1 = bias(layers[1])
    p1 = np.matmul(input_layer, w1) + b1
    a1 = relu(p1)

    wout = weight((layers[1], *output_shape))
    bout = bias(layers[1])
    pout = np.matmul(a1, wout) + bout
    out = relu(pout)

    #  with a bias of zero in each layer
    zero_bias_out = np.prod(input_shape + layers)
    #  with a bias of one in each layer
    one_bias_out = (((input_shape[0]+1) * layers[0]) + 1) * (layers[1]) + 1

    print('input {}'.format(input))
    print('input layer {}'.format(input_layer))
    print('layer 1 {}'.format(a1))
    print('output layer {}'.format(out))

    print('out {}'.format(out))
    print('zero bias out {}'.format(zero_bias_out))
    print('one bias out {}'.format(one_bias_out))

    #  now we make a Qfunc to test
    config['w_init'] = tf.ones
    config['b_init'] = tf.ones

    with tf.Session() as sess:
        q = Qfunc(config)
        sess.run(tf.global_variables_initializer())
        q_vals = sess.run(q.q_values, {q.observation: input.reshape(1, *input_shape)})

    assert q_vals[0].all() == out[0].all()
