import random
import tensorflow as tf
import numpy as np


class QLearn:
    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=0.9):


        # exploration factor between 0-1 (chance of taking a random action)
        self.epsilon = epsilon
        # learning rate between 0-1 (0 means never update Q-values)
        self.alpha = alpha
        # discount factor between 0-1 (higher means the algorithm looks farther into the future
        # at 1 infinite rewards possible -> dont go to 1)
        self.gamma = gamma

        self.actions = actions

        tf.reset_default_graph()

        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs1 = tf.placeholder(shape=[1, len(actions)], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([len(actions), len(actions)], 0, 0.01))
        self.Qout = tf.matmul(self.inputs1, self.W)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, len(actions)], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        self.updateModel = self.trainer.minimize(self.loss)

        self.sess = tf.Session()


    # get the rewards for all actions in a state
    #def getQ(self, state):
    #    # Obtain the Q' values by feeding the new state through our network
    #    Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: np.identity(self.actions)[state:state + 1]})
    #    return Q1

    # use Q-learning formula to update nn when an action is taken
    def learn(self, state, action, reward, newstate):
        # Obtain the Q' values by feeding the new state through our network
        Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: np.identity(len(self.actions))[newstate:newstate + 1]})
        # Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)
        targetQ = self.allQ
        targetQ[0, action[0]] = reward + self.gamma * maxQ1
        # Train our network using target and predicted Q values
        _, W1 = self.sess.run([self.updateModel, self.W],
                              feed_dict={self.inputs1: np.identity(len(self.actions))[state:state + 1],
                                         self.nextQ: targetQ})



        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        # documented reward += learning rate * (newly received reward + max possible reward for next state - doc reward)

        # alpha ... learning rate between 0-1 (0 means never update Q-values)
        # maxq ... highest reward for any action done in the new state = max(Q(s',a') (for any action in that mew state)

        #maxq = max([self.getQ(newstate, a) for a in self.actions])

        #qTarget = self.alpha * (reward + self.gamma * maxq)

        # The nn returns a Q value for each action that could be taken in the new state
        # the best action = the highest Q value represents how good the current state is
        # add to that the reward that was received for entering that state and you have the states Q-value

    # returns the best action based on knowledge in nn
    # chance to return a random action = self.epsilon
    def chooseAction(self, state):
        # Choose an action by greedily (with e chance of random action) from the Q-network
        # epsilon = chance to choose a random action
        if random.random() < self.epsilon:
            action = random.choice(len(self.actions))
        else:
            a, self.allQ = self.sess.run([self.predict, self.Qout], feed_dict={self.inputs1: np.identity(len(self.actions))[state:state + 1]})
            action = a[0]
        return action
