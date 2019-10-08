import lgbm
import numpy as np


class QLearn:
    def __init__(self, puzzleSize, epsilon, alpha=0.2, gamma=0.9):

        self.puzzleSize = puzzleSize
        self.actionsSize = puzzleSize ** 2
        self.inputSize = self.actionsSize ** 2

        self.lgbm = lgbm.Solver(self.actionsSize, alpha, gamma)

    # transform state representation using numbers from 0 to N^2-1 to representation using a vector on length N^2
    # for each cell: for N = 2 solution state [[1,2],[3,0]] looks like [[[0,1,0,0],[0,0,1,0]],[[0,0,0,1],[1,0,0,0]]]
    # which is simply represented as [0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0] and used as input for the NN
    def transform_state(self, state):
        rep = np.full(self.inputSize, 0)
        count = 0
        for y in range(self.puzzleSize):
            for x in range(self.puzzleSize):
                num = state[y][x]
                rep[count * self.actionsSize + num] = 1
                count += 1
        return rep

    def chooseAction(self, state):
        state = self.transform_state(state)
        return self.lgbm.act(state)

    def learn(self, state, action, reward, newstate, is_solved, has_moved):
        #if not has_moved:
        #    return
        state = self.transform_state(state)
        newstate = self.transform_state(newstate)
        self.lgbm.remember(state, action, reward, newstate, is_solved)
        if is_solved:
            self.lgbm.experience_replay()

    def get_exploration_rate(self):
        return self.lgbm.exploration_rate