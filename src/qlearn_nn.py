import random
import tensorflow as tf
import numpy as np



class QLearn:
    def __init__(self, puzzleSize, epsilon=0.05, alpha=0.1, gamma=0.9):


        # exploration factor between 0-1 (chance of taking a random action)
        self.epsilon = epsilon
        # learning rate between 0-1 (0 means never update Q-values)
        self.alpha = alpha
        # discount factor between 0-1 (higher means the algorithm looks farther into the future
        # at 1 infinite rewards possible -> dont go to 1)
        self.gamma = gamma

        self.puzzleSize = puzzleSize
        self.actionsSize = puzzleSize**2
        self.actions = range(self.actionsSize)

        tf.reset_default_graph()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init)
        self.sess.run(init_l)

        # These lines establish the feed-forward part of the network used to choose actions
        #self.inputs1 = tf.placeholder(shape=[1, self.actionsSize], dtype=tf.float32)
        self.inputs1 = tf.placeholder(shape=[1, self.actionsSize**2], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([self.actionsSize**2, self.actionsSize], 0, 0.01))
        self.sess.run(self.W.initializer)
        self.Qout = tf.matmul(self.inputs1, self.W)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, self.actionsSize], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        self.updateModel = self.trainer.minimize(self.loss)


    # transform state representation using numbers from 0 to N^2-1 to representation using a vector on length N^2
    # for each cell: for N = 2 solution state [[1,2],[3,0]] looks like [[[0,1,0,0],[0,0,1,0]],[[0,0,0,1],[1,0,0,0]]]
    # which is simply represented as [0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0] and used as input for the NN
    def transformState(self, state):
        rep = np.full((self.actionsSize ** 2), 0)
        count = 0
        for y in range(self.puzzleSize):
            for x in range(self.puzzleSize):
                num = state[y][x]
                rep[count * self.actionsSize + num] = 1
                count += 1
        return rep



    # TODO learning after every step is SLOW
    # use Q-learning formula to update nn when an action is taken
    def learn(self, state, action, reward, newstate):
        #oneD_state = np.asarray(state).flatten()
        oneD_state = self.transformState(state)
        #oneD_newstate = np.asarray(newstate).flatten()
        oneD_newstate = self.transformState(newstate)
        # Obtain the Q' values by feeding the new state through our network
        #Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: np.identity(self.actionsSize)[newstate:newstate + 1]})
        Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: [oneD_newstate]})
        # Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)
        targetQ = self.allQ
        # TODO for some reasong targetQ and thus allQ goes towards infinity FAST -> becomes nan
        targetQ[0, action] = reward + self.gamma * maxQ1
        print("target")
        print(targetQ)
        print(reward)
        print(self.gamma)
        print(maxQ1)
        # Train our network using target and predicted Q values
        _, W1 = self.sess.run([self.updateModel, self.W], feed_dict={self.inputs1: [oneD_state], self.nextQ: targetQ})



        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        # documented reward += learning rate * (newly received reward + max possible reward for next state - doc reward)

        # alpha ... learning rate between 0-1 (0 means never update Q-values)
        # maxq ... highest reward for any action done in the new state = max(Q(s',a') (for any action in that mew state)

        #maxq = max([self.getQ(newstate, a) for a in self.actions])

        #qTarget = self.alpha * (reward + self.gamma * maxq)

        # The nn returns a Q value for each action that could be taken in the new state
        # the best action = the highest Q value represents how good the current state is
        # add to that the reward that was received for entering that state and you have the states Q-value

    # TODO choosing an action is also slow.. not sure if I can do anything about that tho
    # returns the best action based on knowledge in nn
    # chance to return a random action = self.epsilon
    def chooseAction(self, state):
        #oneD_state = np.asarray(state).flatten()
        oneD_state = self.transformState(state)
        # Choose an action by greedily (with e chance of random action) from the Q-network
        # epsilon = chance to choose a random action
        #print(oneD_state)
        a, self.allQ = self.sess.run([self.predict, self.Qout], feed_dict={self.inputs1: [oneD_state]})
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = a[0]
            print(action)
            print(a)
        print("allQ")
        print(self.allQ)
        return action

