import random
#import gym
import numpy as np
from collections import deque

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# from scores.score_logger import ScoreLogger
# from sklearn.model_selection import train_test_split

#ENV_NAME = "CartPole-v1"

#GAMMA = 0.95
#LEARNING_RATE = 0.001

#MEMORY_SIZE = 1000
MEMORY_SIZE = 5000
TEMP_MEMORY_SIZE = 100
#BATCH_SIZE = 20
BATCH_SIZE = 200

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.05
EXPLORATION_DECAY = 0.96


class Solver:

    def __init__(self, action_space, alpha, gamma):
        #self.exploration_rate = epsilon

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.last_memory = deque(maxlen=TEMP_MEMORY_SIZE)

        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=1000, n_jobs=-1))
        self.isFit = False

        #self.alpha = alpha
        self.gamma = gamma

        self.exploration_rate = EXPLORATION_MAX

    def remember(self, state, action, reward, next_state, done):
        self.last_memory.append((state, action, reward, next_state, done))
        #self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        if self.isFit == True:
            q_values = self.model.predict(state.reshape(1,-1))
        else:
            q_values = np.zeros(self.action_space).reshape(1, -1)
        return np.argmax(q_values[0])

    def experience_replay(self):
        self.memory.extend(self.last_memory)
        self.last_memory.clear()
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, int(len(self.memory) / 1))
        X = []
        targets = []
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                if self.isFit:
                    #q_update = (reward + self.gamma * np.amax(self.model.predict(state_next)[0]))
                    q_update = (reward + self.gamma * np.amax(self.model.predict(state_next.reshape(1,-1))))
                else:
                    q_update = reward
            if self.isFit:
                q_values = self.model.predict(state.reshape(1,-1))
            else:
                q_values = np.zeros(self.action_space).reshape(1, -1)
            q_values[0][action] = q_update

            #X.append(list(state[0]))
            X.append(list(state))
            targets.append(q_values[0])
        self.model.fit(X, targets)
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
