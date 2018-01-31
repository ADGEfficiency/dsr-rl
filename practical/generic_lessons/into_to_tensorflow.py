import gym
import numpy as np
import tensorflow as tf

#  two tuples define the shape of the input and output to the network
#  these are essentially the lengths of the numpy arrays
input_shape = (4,)
output_shape = (4,)

#  tensorflow uses placeholders to feed data into the network
#  the placeholder is fed using numpy arrays
#  the first dimension is the number of samples in the batch 
#  we use None to be able to input any batch size we want

#  the second dimension is the shape of one input array
network_input = tf.placeholder(tf.float32, shape=(None, *input_shape), name='input')
 
#  tensorflow uses a Variable object to hold variables which are changed
#  by the model (ie they are a part of the tensorflow model)

#  below we hold weights in a Variable object. We also initalize the
#  variable wth a tensorflow function
#  note how we unpack the shape tuples into the varible initializer
weights = tf.Variable(tf.random_normal([*input_shape,
                                        *output_shape]))

#  use a similar pattern for biases - note we only need the output shape
#  the output shape is essentially the number of nodes in the network
bias = tf.Variable(tf.random_normal([*output_shape]))

#  now we form the network layer using the weights and biases
pre_activation = tf.add(tf.matmul(network_input, weights), bias)
network_output = tf.nn.relu(pre_activation)
#  we can now predictio from out network using the out operation
#  the machinery for learning is below

#  to train we need some sort of target or y_train variable
#  we use a placeholder for this, as the input will come from outside the
#  network
target = tf.placeholder(tf.float32, shape=(None, *output_shape), name='target')

#  the error is the difference between the target and whatever our
#  network is outputting
error = tf.subtract(target, network_output)
#  for the loss function we use the sum of the error squared
loss = tf.reduce_mean(tf.square(error))
#  get an Adam optimizer to do the heavy lifting for us 
#  the learning rate we use here is one of the most important assumptons
#  you make in training a neural network
optimizer = tf.train.AdamOptimizer(0.001)
#  the operation to train the network is to use our optimier to minimize
#  the loss across the batch
train_op = optimizer.minimize(loss)

#  here we use the summary functionality of tensorflow to see whats going
#  on in the nework
tf.summary.tensor_summary('input', network_input)
tf.summary.histogram('weights', weights)
tf.summary.histogram('error', error)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

def generate_data(num_samples):
    """
    Generates training data for our network.

    args
        num_samples (int)

    returns
        net_in (np.array)  fed into the network (aka features or x)
        net_out (np.array)  aka target or y - the value we are trying to approx.
    """
    net_in = np.random.rand(num_samples, *input_shape)
    net_out = np.square(net_in)

    return net_in, net_out


#  run the function to get data to train with
inputs, targets = generate_data(1000)

#  a key paradigm in tensorflow is the Session object
#  you can think of the tensorflow session as a tf model
with tf.Session() as sess:
    #  a necessary bit if boiler plate in tensorflow - we have to
    #  initialize all our variables.  This sets the initial values of our
    #  weights and biases
    sess.run(tf.global_variables_initializer())

    #  the writer object is what works with tensorboard
    writer = tf.summary.FileWriter('./logs', graph=sess.graph)

    #  lets do some training
    #  here we do 10000 batches of training
    #  but we are sending in the entire dataset as the batch
    for train_step in range(10000):
        #  now we run the tensorflow graph
        #  the graph is run by calling the .run method on the session
        #  the run method takes two inputs:
        #   fetches = the tf operations to run
        #   feed_dict = values for the placeholders

        #  here we fetch two operations
        #   train_op - the operation to train the network
        #   summary - the summary operations for the graph
        fetches = [loss, train_op, merged]

        #  the feed_dict is a dictionary with
        #   keys = the placeholders
        #   values = the numpy arrays
        #   note that we feed in multiple samples
        feed_dict = {network_input: inputs,
                     target: targets}
        print(train_step)
        #  finally we run the session using the fetches and feed_dict
        loss_value, _, summary = sess.run(fetches, feed_dict)
        #  the operation to add the summary to the tensorboard output file
        writer.add_summary(summary, train_step)
        print('step {} loss {}'.format(train_step, loss_value))

    #  generate a test set 
    test_in, test_out = generate_data(1)

    #  here we get predictions from our network - we don't train
    pred = sess.run(network_output, {network_input: test_in})
    print(pred)
    print(test_out)
