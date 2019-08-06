import random


class QLearn:
    def __init__(self, puzzleSize, epsilon=0.1, alpha=0.2, gamma=0.9):
        # q is a dictionary
        self.q = {}

        # exploration factor between 0-1 (chance of taking a random action)
        self.epsilon = epsilon
        # learning rate between 0-1 (0 means never update Q-values)
        self.alpha = alpha
        # discount factor between 0-1 (higher means the algorithm looks farther into the future
        # at 1 infinite rewards possible -> dont go to 1)
        self.gamma = gamma

        self.actions = range(puzzleSize ** 2)

        if puzzleSize == 2:
            self.batchMaxSize = 10
        if puzzleSize == 3:
            self.batchMaxSize = 200

        self.batch = []
        self.batchSize = 0
        self.stepsSinceLastLearned = 0

    # TODO does this cost too much time??
    # turns a state (list of lists) into a tuple of tuples so dict can handle it as a key
    def turnStateIntoTuple(self, state):
        return tuple(tuple(i) for i in state)

    # get the reward for a (state,action) pair
    # if there is no entry in the dict yet, return 0 (or 1)
    def getQ(self, state, action):
        stateTuple = self.turnStateIntoTuple(state)
        return self.q.get((stateTuple, action), 0.0)
        # return self.q.get((state, action), 1.0)

    # use Q-learning formula to update old (s,a):r entries in the dict when an action is taken
    def learnQ(self, state, action, reward, maxq):
        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        # documented reward += learning rate * (newly received reward + max possible reward for next state - doc reward)

        # maxq ... highest reward for any action done in the new state (takes discount gamma into account)
        # alpha ... learning rate between 0-1 (0 means never update Q-values)

        stateTuple = self.turnStateIntoTuple(state)

        # oldreward is the reward for (s,a) as it was before
        oldreward = self.q.get((stateTuple, action), None)

        if (oldreward is not None) & (maxq is not None):
            # if there was already a value, update it according to the Q-learning algorithm
            self.q[(stateTuple, action)] = oldreward + self.alpha * (reward + maxq - oldreward)
        elif maxq is not None:
            # if tuple seen for the first time, but some neighbour state is known -> use reward and best next state
            self.q[(stateTuple, action)] = reward + maxq
        else:
            # if tuple seen for the first time and no other information available -> use just current reward
            self.q[(stateTuple, action)] = reward

    # returns the best action based on knowledge in q dictionary
    # chance to return a random action = self.epsilon
    def chooseAction(self, state):
        # epsilon = chance to choose a random action
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            # In case there're several state-action max values 
            # we select a random one among them
            if q.count(maxQ) > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state, action, reward, newstate, isSolved, hasMoved):

        self.stepsSinceLastLearned += 1

        if self.batchSize >= self.batchMaxSize:
            self.batch.pop(0)
        else:
            self.batchSize += 1
        self.batch.append([state, action, reward, newstate])

        if isSolved:
            chosenBatch = self.batch[::-1]
            for i in range(len(chosenBatch)):
                b = chosenBatch[i]
                if b[3] is not None:
                    maxqnew = max([self.getQ(b[3], a) for a in self.actions])
                    maxqnew *= self.gamma
                else:
                    maxqnew = None
                self.learnQ(b[0], b[1], b[2], maxqnew)
            self.batch=[]
            self.batchSize=0
        else:
            if newstate is not None:
                maxqnew = max([self.getQ(newstate, a) for a in self.actions])
                maxqnew *= self.gamma
            else:
                maxqnew = None
            self.learnQ(state, action, reward, maxqnew)