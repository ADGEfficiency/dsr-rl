import tensorflow as tf


class Qfunc(object):
    """
    Approximation of the action-value function Q(s,a) using a feedforward
    neural network built in TensorFlow.

    args
        model_config (dict) used to build tf machinery.  see make_graph for args
        scope (str)

    methods
        make_graph(**model_config)

    attributes
        observation
        target
        q_values
        max_q
        optimal_action_idx
        error
        loss
        train_op
        acting_summary
        learning_summary
    """
    def __init__(self, model_config, scope):
        self.scope = scope

        with tf.variable_scope(scope):
            self.make_graph(**model_config)

        #  tensorflow variables for this model
        #  weights and biases of the neural network 
        params = [p for p in tf.trainable_variables()
                  if p.name.startswith(scope)]

        #  sort parameters list by the variable name 
        self.params = sorted(params, key=lambda var: var.name)

    def __repr__(self): return '<Q(s,a) {} network>'.format(self.scope)

    def make_graph(self,
                   input_shape,
                   output_shape,
                   layers,
                   learning_rate,
                   w_init=tf.truncated_normal,
                   b_init=tf.zeros,
                   **kwargs):
        """
        Creates all Tensorflow machinery required for acting and learning.

        Could be split into a acting & learning section.

        We never need to train our target network - params for the target
        network are updated by copying weights - done in the DQN agent.

        args
            input_shape (tuple)
            output_shape (tuple)
            layers (list)
            learning_rate (float)
            wt_init (function) tf function used to initialize weights
            b_init (function) tf function used to initialize the biases
        """
        print('making tf graph for {}'.format(self.scope))
        print('input shape {}'.format(input_shape))
        print('output shape {}'.format(output_shape))
        print('layers {}'.format(layers))

        #  aka state - the input to the network
        self.observation = tf.placeholder(tf.float32,
                                          shape=(None, *input_shape),
                                          name='observation')

        #  used to index the network output
        #  first dimension = batch
        #  second dimension = the index of the action
        #  [0, 1] = first batch sample, second action
        #  [4, 0] = 5th sample, first action
        self.action = tf.placeholder(tf.int32,
                                      shape=(None, 2),
                                      name='action')

        #  the target is for the action being trained
        #  shape = (batch_size, 1)
        self.target = tf.placeholder(tf.float32,
                                     shape=(None, 1),
                                     name='target')

        with tf.name_scope('input_layer'):
            #  variables for the input layer weights & biases
            w1 = tf.Variable(w_init([*input_shape, layers[0]]), 'in_w')
            b1 = tf.Variable(b_init(layers[0]), 'in_bias')

            #  construct the layer and use a relu at the end
            layer = tf.add(tf.matmul(self.observation, w1), b1)
            layer = tf.nn.relu(layer)

        for i, nodes in enumerate(layers[1:], 1):
            with tf.variable_scope('hidden_layer_{}'.format(i)):
                w = tf.Variable(w_init([layers[i-1], nodes]), '{}_w'.format(i))
                b = tf.Variable(b_init(nodes), '{}_b'.format(i))
                layer = tf.add(tf.matmul(layer, w), b)
                layer = tf.nn.relu(layer)

        with tf.variable_scope('output_layer'):
            wout = tf.Variable(w_init([nodes, *output_shape]), 'out_w')
            bout = tf.Variable(b_init(*output_shape), 'out_b')

            #  no activation function on the output layer (i.e. linear)
            self.q_values = tf.add(tf.matmul(layer, wout), bout)

        with tf.variable_scope('argmax'):
            #  ops for selecting optimal action
            max_q = tf.reduce_max(self.q_values, axis=1, name='max_q')
            self.max_q = tf.reshape(max_q, (-1, 1))
            self.optimal_action_idx = tf.argmax(self.q_values, axis=1,
                                                name='optimal_action_idx')

        with tf.variable_scope('learning'):
            #  index out Q(s,a) for the action being trained
            q_value = tf.gather_nd(self.q_values, self.action, name='q_value')
            self.q_value = tf.reshape(q_value, (-1, 1))

            self.error = self.target - self.q_value
            self.loss = tf.losses.huber_loss(self.target, self.q_value)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        #  averages across the batch (ie a scalar to represent the whole batch)
        average_q_val = tf.reduce_mean(self.q_value)

        acting_sum = [tf.summary.histogram('observation_act', self.observation),
                      tf.summary.histogram('input_weights', w1),
                      tf.summary.histogram('input_bias', b1),
                      tf.summary.histogram('output_weights', wout),
                      tf.summary.histogram('output_bias', bout),
                      tf.summary.histogram('q_values', self.q_values),
                      tf.summary.histogram('max_q', self.max_q)]

        self.acting_summary = tf.summary.merge(acting_sum)

        learn_sum = [tf.summary.histogram('observation_learn', self.observation),
                     tf.summary.histogram('q_values', self.q_values),
                     tf.summary.histogram('max_q', self.max_q),
                     tf.summary.histogram('target', self.target),
                     tf.summary.scalar('avg_batch_q_value', average_q_val),
                     tf.summary.histogram('error', self.error),
                     tf.summary.scalar('loss', self.loss)]

        self.learning_summary = tf.summary.merge(learn_sum)
