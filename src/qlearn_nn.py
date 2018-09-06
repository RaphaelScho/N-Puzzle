import random
import tensorflow as tf
import numpy as np



class QLearn:
    def __init__(self, puzzleSize, epsilon=0.5, alpha=0.33, gamma=0.9):


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
        self.inputSize = self.actionsSize**2

        if self.puzzleSize == 2:
            self.hiddenLayerSize = self.inputSize**2
        elif self.puzzleSize == 3:
            self.hiddenLayerSize = self.inputSize**1.5 # TODO not set yet .. 2 makes it reaaally slow...
        elif self.puzzleSize == 4:
            self.hiddenLayerSize = self.inputSize**1.4 # TODO not set yet

        tf.reset_default_graph()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init)
        self.sess.run(init_l)

        # These lines establish the feed-forward part of the network used to choose actions
        ##self.inputs1 = tf.placeholder(shape=[1, self.actionsSize], dtype=tf.float32)
        #self.inputs1 = tf.placeholder(shape=[1, self.actionsSize**2], dtype=tf.float32)
        #self.W = tf.Variable(tf.random_uniform([self.actionsSize**2, self.actionsSize], 0, 0.01))
        #self.sess.run(self.W.initializer)
        #self.Qout = tf.matmul(self.inputs1, self.W)
        #self.predict = tf.argmax(self.Qout, 1)

        # hidden layers instead
        self.inputs1 = tf.placeholder(shape=[1,self.inputSize], dtype=tf.float32)
        fc1 = tf.layers.dense(self.inputs1, self.hiddenLayerSize, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, self.hiddenLayerSize, activation=tf.nn.relu)
        self.Qout = tf.layers.dense(fc2, self.actionsSize)
        self.predict = tf.argmax(self.Qout, 1)


        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, self.actionsSize], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(tf.clip_by_value(self.nextQ - self.Qout,-2,2)))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        self.updateModel = self.trainer.minimize(self.loss)
        self._var_init = tf.global_variables_initializer()
        self.sess.run(self._var_init)

        self.batch = []
        self.batchSize = 0
        # TODO those values might also need to change based on puzzle size
        self.maxBatchSize = 5000 # how many [state,action,reward,newstate] tuples to remember
        self.learningSteps = 1000  # after how many actions should a batch be learned
        self.learnSize = 1500 # how many of those tuples to randomly choose when learning
        self.age = 0



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

    def doLearning(self, oneD_state, action, reward, oneD_newstate):
        # Obtain the Q' values by feeding the new state through our network
        # Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: np.identity(self.actionsSize)[newstate:newstate + 1]})
        allQ = self.sess.run(self.Qout, feed_dict={self.inputs1: [oneD_state]})
        targetQ = allQ
        #maxQ1 = 0
        if oneD_newstate is not None:
            Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1: [oneD_newstate]})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.amax(Q1)
            targetQ[0, action] = reward + self.gamma * maxQ1
        else:
            targetQ[0, action] = reward

        # Train our network using target and predicted Q values
        # _, W1 = self.sess.run([self.updateModel, self.W], feed_dict={self.inputs1: [oneD_state], self.nextQ: targetQ})
        self.sess.run(self.updateModel, feed_dict={self.inputs1: [oneD_state], self.nextQ: targetQ})

        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        # documented reward += learning rate * (newly received reward + max possible reward for next state - doc reward)

        # alpha ... learning rate between 0-1 (0 means never update Q-values)
        # maxq ... highest reward for any action done in the new state = max(Q(s',a') (for any action in that mew state)

        # maxq = max([self.getQ(newstate, a) for a in self.actions])

        # qTarget = self.alpha * (reward + self.gamma * maxq)

        # The nn returns a Q value for each action that could be taken in the new state
        # the best action = the highest Q value represents how good the current state is
        # add to that the reward that was received for entering that state and you have the states Q-value

    # use Q-learning formula to update nn when an action is taken
    def learn(self, state, action, reward, newstate):
        #oneD_state = np.asarray(state).flatten()
        oneD_state = self.transformState(state)
        #oneD_newstate = np.asarray(newstate).flatten()
        if newstate is not None:
            oneD_newstate = self.transformState(newstate)
        else:
            oneD_newstate = None

        if self.batchSize > self.maxBatchSize:
            self.batch.pop(0)
        else:
            self.batchSize += 1
        self.batch.append([oneD_state, action, reward, oneD_newstate])

        # TODO it uses less space to store states in original form and only transform when chosen,
        # increases calc time though since states are chosen 1.5 times on average

        if self.age % self.learningSteps == 0:
            #random.shuffle(self.batch)
            if self.batchSize < self.learnSize:
                chosenBatch = random.sample(self.batch, self.batchSize)
            else:
                chosenBatch = random.sample(self.batch, self.learnSize)
            for i in range(len(chosenBatch)):
                b = chosenBatch[i]
                self.doLearning(b[0],b[1],b[2],b[3])

    # returns the best action based on knowledge in nn
    # chance to return a random action = self.epsilon
    def chooseAction(self, state):
        self.age += 1
        # Choose an action by greedily (with e chance of random action) from the Q-network
        # epsilon = chance to choose a random action
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            oneD_state = self.transformState(state)
            a, allQ = self.sess.run([self.predict, self.Qout], feed_dict={self.inputs1: [oneD_state]})
            action = a[0]
            #if self.epsilon < 0.05:
            #    print("action %f" %(action))
            #    print("allQ")
            #    print(allQ)
        return action

