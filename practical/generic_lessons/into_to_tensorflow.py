import gym
import numpy as np
import tensorflow as tf
input_shape = (4,)
output_shape = (4,)

input = tf.placeholder(tf.float32, shape=(None, *input_shape), name='input')
target = tf.placeholder(tf.float32, shape=(None, *output_shape), name='target')

weights = tf.Variable(tf.random_normal([*input_shape,
                                        *output_shape]))

bias = tf.Variable(tf.random_normal([*output_shape]))

out = tf.add(tf.matmul(input, weights), bias)

error = tf.subtract(target, out)
optimizer = tf.train.AdamOptimizer(0.001)
loss = tf.reduce_sum(tf.square(error))
train_op = optimizer.minimize(loss)

tf.summary.tensor_summary('input', input)
tf.summary.histogram('weights', weights)
tf.summary.histogram('error', error)
tf.summary.scalar('loss', loss)

merged = tf.summary.merge_all()

def generate_data(num_samples):
    input = np.random.rand(num_samples, *input_shape)
    
    output = np.square(input)

    return input, output

inputs, targets = generate_data(1000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./logs', graph=sess.graph)
    for i in range(10000):
        summary, _ = sess.run([merged, train_op], {input: inputs,
                                                          target: targets})
        writer.add_summary(summary, i)
    
    test_in, test_out = generate_data(1)

    pred = sess.run(out, {input: test_in}) 
    print(pred)
    print(test_out)
