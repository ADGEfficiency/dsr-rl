import tensorflow as tf

class Qfunc(object):
    """

    args
        config (dict)
            input_shape (tuple)
            output_shape (tuple)
            weights (list)
    methods
        make_graph()

    attributes

        observation
        target
        q_values
        max_q
        optimal_action_idx
        error
        loss
        train_op
        all_summaries
    """
    def __init__(self, config):
        self.make_graph(**config)

    def make_graph(self, input_shape, output_shape, layers):

        weight_init = tf.truncated_normal
        bias_init = tf.zeros

        with tf.variable_scope('DQN'):

            self.observation = tf.placeholder(tf.float32,
                                              shape=(None, *input_shape),
                                              name='observation')

            self.target = tf.placeholder(tf.float32,
                                         shape=(None, *output_shape),
                                         name='target')

            with tf.variable_scope('input_layer'):
                w1 = tf.Variable(weight_init([*input_shape, layers[0]]))
                b1 = tf.Variable(bias_init(layers[0]))

                layer = tf.add(tf.matmul(self.observation, w1), b1)
                layer = tf.nn.relu(layer)

            for i, nodes in enumerate(layers[1:], 1):
                with tf.variable_scope('hidden_layer'.format(i)):
                    w = tf.Variable(weight_init(*[layers[i-1], nodes]))
                    b = tf.Variable(bias_init(*layers[i]))

                    layer = tf.add(tf.matmul(layer, w), b)
                    layer = tf.nn.relu(layer)

            with tf.variable_scope('output_layer'):
                wout = tf.Variable(weight_init([*nodes, *output_shape]))
                bout = tf.Variable(bias_init(*output_shape))
                self.q_values = tf.add(tf.matmul(layer, wout), bout)

            self.max_q = tf.reduce_max(self.q_values, axis=1, name='max_Q')
            self.optimal_action_idx = tf.argmax(self.q_values, axis=1)

            self.error = tf.square(tf.subtract(self.target, self.q_values))
            self.loss = tf.losses.mean_squared_error(self.target, self.q_values)
            optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = optimizer.minimize(self.loss)

            tf.summary.histogram('input_weights', w1)
            tf.summary.histogram('output_weights', wout)
            tf.summary.histogram('error', self.error)
            tf.summary.scalar('loss', self.loss)

            self.all_summaries = tf.summary.merge_all()


