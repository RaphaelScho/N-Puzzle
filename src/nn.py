import tensorflow as tf


class nn:
    def __init__(self, puzzleSize, alpha):
        self.alpha = alpha
        self.puzzleSize = puzzleSize
        self.actionsSize = puzzleSize ** 2
        self.inputSize = self.actionsSize ** 2

        self.hiddenLayerSize = None
        self.hiddenLayerSize2 = None

        if puzzleSize == 2:
            self.hiddenLayerSize = round(self.inputSize / 2)
        if puzzleSize == 3:
            self.hiddenLayerSize = round(self.inputSize / 1)
            self.hiddenLayerSize2 = round(self.inputSize / 2)

        tf.reset_default_graph()

        # layers
        self.inputs = tf.placeholder(shape=[1, self.inputSize], dtype=tf.float32)
        fc1 = self.fc_layer(self.inputs, self.hiddenLayerSize, "L1", use_relu=True)
        if self.hiddenLayerSize2 is not None:
            fc2 = self.fc_layer(fc1, self.hiddenLayerSize2, "L2", use_relu=True)
            self.Qout = tf.layers.dense(fc2, 1)
        else:
            self.Qout = tf.layers.dense(fc1, 1)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, 1], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(tf.clip_by_value(self.nextQ - self.Qout, -2, 2)))  # TODO try -10, 10
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        # self.trainer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.updateModel = self.trainer.minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)


    # helper functions below from https://easy-tensorflow.com/tf-tutorials/neural-networks/two-layer-neural-network
    # (from 06.08.2019)

    # weight and bais wrappers
    def weight_variable(self, name, shape):
        """
        Create a weight variable with appropriate initialization
        :param name: weight name
        :param shape: weight shape
        :return: initialized weight variable
        """
        initer = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable('W_' + name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=initer)

    def bias_variable(self, name, shape):
        """
        Create a bias variable with appropriate initialization
        :param name: bias variable name
        :param shape: bias variable shape
        :return: initialized bias variable
        """
        initial = tf.constant(0., shape=shape, dtype=tf.float32)
        return tf.get_variable('b_' + name,
                               dtype=tf.float32,
                               initializer=initial)

    def fc_layer(self, x, num_units, name, use_relu=True):
        """
        Create a fully-connected layer
        :param x: input from previous layer
        :param num_units: number of hidden units in the fully-connected layer
        :param name: layer name
        :param use_relu: boolean to add ReLU non-linearity (or not)
        :return: The output array
        """
        in_dim = x.get_shape()[1]
        W = self.weight_variable(name, shape=[in_dim, num_units])
        b = self.bias_variable(name, [num_units])
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer
