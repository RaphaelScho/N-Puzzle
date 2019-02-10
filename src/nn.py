import tensorflow as tf


class nn:
    def __init__(self, puzzleSize, alpha):
        self.alpha = alpha
        self.puzzleSize = puzzleSize
        self.actionsSize = puzzleSize ** 2
        self.inputSize = self.actionsSize ** 2

        #self.hiddenLayerSize = self.inputSize ** 0.7  # **1.5 # TODO experimental
        self.hiddenLayerSize = 10#21 # TODO experimental

        tf.reset_default_graph()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init)
        self.sess.run(init_l)

        # layers
        self.inputs = tf.placeholder(shape=[1, self.inputSize], dtype=tf.float32)
        fc1 = tf.layers.dense(self.inputs, self.hiddenLayerSize, activation=tf.nn.relu)
        # fc2 = tf.layers.dense(fc1, self.hiddenLayerSize, activation=tf.nn.relu)
        self.Qout = tf.layers.dense(fc1, 1)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, 1], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(tf.clip_by_value(self.nextQ - self.Qout, -2, 2)))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        self.updateModel = self.trainer.minimize(self.loss)
        self._var_init = tf.global_variables_initializer()
        self.sess.run(self._var_init)
