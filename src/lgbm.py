import random
#import gym
import numpy as np
from collections import deque
import copy

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# from scores.score_logger import ScoreLogger
# from sklearn.model_selection import train_test_split

#ENV_NAME = "CartPole-v1"

#GAMMA = 0.95
#LEARNING_RATE = 0.001

#MEMORY_SIZE = 1000
# selfMEMORY_SIZE = 2500
# selfTEMP_MEMORY_SIZE = 300
# selfMIN_BATCH_SIZE = 500

#selfEXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.05
EXPLORATION_DECAY = 0.985


class Solver:

    def __init__(self, puzzle_size, epsilon, alpha, gamma):
        #self.exploration_rate = epsilon

        if puzzle_size == 2:
            self.MEMORY_SIZE = 1550
            self.TEMP_MEMORY_SIZE = 50
            self.MIN_BATCH_SIZE = 1550
            max_depth = 9
            n_estimators = 50
            self.EXPLORATION_MAX = epsilon
        #elif puzzle_size == 3:
        else:
            self.MEMORY_SIZE = 5500
            self.TEMP_MEMORY_SIZE = 1000
            self.MIN_BATCH_SIZE = 500
            max_depth = 20
            n_estimators = 200
            self.EXPLORATION_MAX = epsilon

        #self.action_size = puzzle_size ** 2
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.last_memory = deque(maxlen=self.TEMP_MEMORY_SIZE)

        self.actions = range(4)

        # create one nn per action:
        self.models = {}
        for action in self.actions:
            model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, num_leaves=2^max_depth, subsample_for_bin = self.MEMORY_SIZE, n_jobs=-1, learning_rate=alpha, min_split_gain=0.04)
            self.models[action] = copy.copy(model)

        #self.model = LGBMRegressor(n_estimators=100, num_leaves=35, max_depth=5, subsample_for_bin = round(MEMORY_SIZE * 0.3), n_jobs=-1, learning_rate=0.1)
        self.isFit = False

        #self.alpha = alpha
        self.gamma = gamma

        self.exploration_rate = self.EXPLORATION_MAX
        self.mem_count = 0

    def remember(self, state, action, reward, next_state, done):
        self.last_memory.append((state, action, reward, next_state, done))
        #self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(len(self.actions))
        if self.isFit:
            q_values = []
            for action in self.actions:
                q_value = self.models[action].predict(state.reshape(1,-1))[0]
                q_values.append(q_value)
            #q_values = self.model.predict(state.reshape(1,-1))
        else:
            q_values = np.zeros(len(self.actions)).reshape(1, -1)[0]
        return np.argmax(q_values)

    def experience_replay(self):
        # size = len(self.memory)
        self.memory.extend(self.last_memory)
        self.mem_count += len(self.last_memory)
        if self.mem_count > self.MEMORY_SIZE:
            print("-------------------------------------")
            print("Max memory size reached (" + str(self.MEMORY_SIZE) + ")")
            print("-------------------------------------")
            self.mem_count = 0
        self.last_memory.clear()
        if len(self.memory) < self.MIN_BATCH_SIZE:
            return
        batch = random.sample(self.memory, int(len(self.memory) / 1))
        X = {}
        targets = {}
        for action in self.actions:
            targets[action] = []
            X[action] = []
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                if self.isFit:
                    q_values = []
                    for act in self.actions:
                        q_values.append(reward + self.gamma * np.amax(self.models[act].predict(state_next.reshape(1,-1))))
                    q_update = np.max(q_values)
                else:
                    q_update = reward

            X[action].append(list(state))
            targets[action].append(q_update)
        for action in self.actions:
            self.models[action].fit(X[action], targets[action])
        self.isFit = True
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


# def cartpole():
#     env = gym.make(ENV_NAME)
#     # score_logger = ScoreLogger(ENV_NAME)
#     observation_space = env.observation_space.shape[0]
#     action_space = env.action_space.n
#     solver = Solver(observation_space, action_space)
#     run = 0
#     while True:
#         run += 1
#         state = env.reset()
#         state = np.reshape(state, [1, observation_space])
#         step = 0
#         while True:
#             step += 1
#             # env.render()
#             action = solver.act(state)
#             state_next, reward, terminal, info = env.step(action)
#             reward = reward if not terminal else -reward
#             state_next = np.reshape(state_next, [1, observation_space])
#             solver.remember(state, action, reward, state_next, terminal)
#             state = state_next
#             if terminal:
#                 print(
#                     "Run: " + str(run) + ", exploration: " + str(solver.exploration_rate) + ", score: " + str(step))
#                 # score_logger.add_score(step, run)
#                 break
#         solver.experience_replay()


# if __name__ == "__main__":
#     cartpole()
