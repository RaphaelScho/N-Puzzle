import random
import numpy as np
import nn
import copy


class QLearn:
    def __init__(self, puzzleSize, epsilon, alpha, gamma):

        # exploration factor between 0-1 (chance of taking a random action)
        self.epsilon = epsilon
        # learning rate between 0-1 (0 means never update Q-values, 1 means discard old value)
        self.alpha = alpha
        # discount factor between 0-1 (higher means the algorithm looks farther into the future
        # at 1 infinite rewards possible -> dont go to 1)
        self.gamma = gamma

        self.puzzleSize = puzzleSize
        self.actionsSize = puzzleSize**2
        self.actions = range(self.actionsSize)
        self.inputSize = self.actionsSize**2

        # create one nn per action:
        self.networks = {}
        for action in self.actions:
            net = nn.nn(puzzleSize = self.puzzleSize, alpha = self.alpha)
            self.networks[action] = copy.copy(net)

        if puzzleSize == 2:
            self.batchMaxSize = 50
            self.moveBatchMaxSize = 10
            self.learningSteps = 20
            self.learnSize = 10
        if puzzleSize == 3:
            self.batchMaxSize = 450  # how many [state,action,reward,newstate] tuples to remember
            self.moveBatchMaxSize = 50 # how many tuples to remember where a change in the boardstate happened
            self.learningSteps = 200  # after how many actions should a batch be learned
            self.learnSize = 20  # how many of those tuples to randomly choose when learning

        self.age = 0
        self.batch = []
        self.batchSize = 0
        self.moveBatch = []
        self.moveBatchSize = 0
        self.winBatch = []
        #self.chosenActions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    # transform state representation using numbers from 0 to N^2-1 to representation using a vector on length N^2
    # for each cell: for N = 2 solution state [[1,2],[3,0]] looks like [[[0,1,0,0],[0,0,1,0]],[[0,0,0,1],[1,0,0,0]]]
    # which is simply represented as [0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0] and used as input for the NN
    def transformState(self, state):
        rep = np.full(self.inputSize, 0)
        count = 0
        for y in range(self.puzzleSize):
            for x in range(self.puzzleSize):
                num = state[y][x]
                rep[count * self.actionsSize + num] = 1
                count += 1
        return rep

    def doLearning(self, oneD_state, action, reward, oneD_newstate):
        # Obtain the Q' values by feeding the new state through our network
        # this was originally because there was one network for all actions, but we only wanted to change one action,
        # so we ran it through and then only changed the targetQ value for the one action (=output)
        #allQ = self.networks[action].sess.run(self.networks[action].Qout,
        #                                      feed_dict={self.networks[action].inputs: [oneD_state]})
        #targetQ = allQ
        targetQ = np.ndarray(shape=(1,1), dtype=float)
        if oneD_newstate is not None:
            # calculate Q1 for all actions and find maximum
            qList = []
            for a in self.actions:
                Q1 = self.networks[a].sess.run(self.networks[a].Qout,
                                                feed_dict={self.networks[a].inputs: [oneD_newstate]})
                #qList.append(np.amax(Q1))
                qList.append(Q1[0][0])
            # Obtain maxQ' and set our target value for chosen action.
            #maxQ1 = np.amax(Q1)
            maxQ1 = max(qList)
            targetQ[0] = reward + self.gamma * maxQ1
        else:
            targetQ[0] = reward

        # Train network using target and predicted Q values
        # _, W1 = self.sess.run([self.updateModel, self.W], feed_dict={self.inputs1: [oneD_state], self.nextQ: targetQ})
        self.networks[action].sess.run(self.networks[action].updateModel,
                                       feed_dict={self.networks[action].inputs: [oneD_state],
                                                  self.networks[action].nextQ: targetQ})

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
    # def learn_(self, state, action, reward, newstate, isSolved, hasMoved):
    #     oneD_state = self.transformState(state)
    #     if newstate is not None:
    #         oneD_newstate = self.transformState(newstate)
    #     else:
    #         oneD_newstate = None
    #
    #     if self.batchSize >= self.batchMaxSize:
    #         self.batch.pop(0)
    #     else:
    #         self.batchSize += 1
    #     self.batch.append([oneD_state, action, reward, oneD_newstate])
    #
    #     if hasMoved:
    #         if self.moveBatchSize >= self.moveBatchMaxSize:
    #             self.moveBatch.pop(0)
    #         else:
    #             self.moveBatchSize += 1
    #         self.moveBatch.append([oneD_state, action, reward, oneD_newstate])
    #
    #     if isSolved:
    #         #chosenBatch = self.batch[:-self.learnSize:-1]
    #         chosenBatch = self.moveBatch[::-1]
    #         self.winBatch = copy.deepcopy(chosenBatch)
    #
    #         #self.batch = []
    #         #self.batchSize = 0
    #         for i in range(len(chosenBatch)):
    #             b = chosenBatch[i]
    #             self.doLearning(b[0], b[1], b[2], b[3])
    #
    #         self.moveBatch = []
    #         self.moveBatchSize = 0
    #
    #     elif self.age % self.learningSteps == 0:
    #         #if self.batchSize < self.learnSize:
    #         chosenBatch = random.sample((self.batch + self.winBatch), min(self.batchSize, self.learnSize))
    #             #chosenBatch = random.sample(self.batch, self.batchSize)
    #         #else:
    #          #   chosenBatch = random.sample((self.batch + self.winBatch), self.learnSize)
    #             #chosenBatch = random.sample(self.batch, self.learnSize)
    #         for i in range(len(chosenBatch)):
    #             b = chosenBatch[i]
    #             self.doLearning(b[0], b[1], b[2], b[3])

    def learn(self, state, action, reward, newstate, isSolved, hasMoved):
        oneD_state = self.transformState(state)
        if newstate is not None:
            oneD_newstate = self.transformState(newstate)
        else:
            oneD_newstate = None

        if self.batchSize >= self.batchMaxSize:
            self.batch.pop(0)
        else:
            self.batchSize += 1
        self.batch.append([oneD_state, action, reward, oneD_newstate])

        if isSolved:
            chosenBatch = self.batch[::-1]
            for i in range(len(chosenBatch)):
                b = chosenBatch[i]
                # if b[3] is not None:
                #     maxqnew = max([self.getQ(b[3], a) for a in self.actions])
                #     maxqnew *= self.gamma
                # else:
                #     maxqnew = None
                self.doLearning(b[0], b[1], b[2], b[3])
            self.batch=[]
            self.batchSize=0
        elif self.age % self.batchMaxSize == 0:
            chosenBatch = self.batch[::-1]
            for i in range(len(chosenBatch)):
                b = chosenBatch[i]
                # if b[3] is not None:
                #     maxqnew = max([self.getQ(b[3], a) for a in self.actions])
                #     maxqnew *= self.gamma
                # else:
                #     maxqnew = None
                self.doLearning(b[0], b[1], b[2], b[3])
            #self.doLearning(oneD_state, action, reward, oneD_newstate)

    # returns the best action based on knowledge in nn
    # chance to return a random action = self.epsilon
    def chooseAction(self, state):
        self.age += 1
        # Choose an action by greedily (with e chance of random action) from the Q-network
        # epsilon = chance to choose a random action
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            # get max of all networks
            oneD_state = self.transformState(state)
            actList = []
            for a in self.actions:
                act, allQ = self.networks[a].sess.run([self.networks[a].predict, self.networks[a].Qout],
                                                      feed_dict={self.networks[a].inputs: [oneD_state]})
                actList.append(allQ[0][0])
            #action = actList.index(max(actList))
            maxval = max(actList)
            maxActions = [i for i, x in enumerate(actList) if x == maxval]
            action = random.choice(maxActions)
        #self.chosenActions[action] += 1
        return action
