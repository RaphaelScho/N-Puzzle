import random


class QLearn:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):


        # exploration factor between 0-1 (chance of taking a random action)
        self.epsilon = epsilon
        # learning rate between 0-1 (0 means never update Q-values)
        self.alpha = alpha
        # discount factor between 0-1 (higher means the algorithm looks farther into the future
        # at 1 infinite rewards possible -> dont go to 1)
        self.gamma = gamma

        self.actions = actions


    # get the reward for a (state,action) pair
    def getQ(self, state, action):
        return 0  # TODO

    # use Q-learning formula to update nn when an action is taken
    def learn(self, state, action, reward, newstate):
        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        # documented reward += learning rate * (newly received reward + max possible reward for next state - doc reward)

        # alpha ... learning rate between 0-1 (0 means never update Q-values)
        # maxq ... highest reward for any action done in the new state = max(Q(s',a') (for any action in that mew state)

        maxq = max([self.getQ(newstate, a) for a in self.actions])

        qTarget = self.alpha * (reward + self.gamma * maxq)


        # The nn returns a Q value for each action that could be taken in the new state
        # the best action = the highest Q value represents how good the current state is
        # add to that the reward that was received for entering that state and you have the states Q-value

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

