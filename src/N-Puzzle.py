import random
import cellular

import qlearn as learner

puzzleSize = 3  # Size 3 means 3x3 puzzle
puzzleActionSize = puzzleSize ** 2 # number of possible actions given puzzleSize
puzzleCells = range(puzzleActionSize) # list of actions ( = tiles ) in the puzzle


# creates random puzzle based on puzzleSize
def createRandomPuzzle():
    pass
    # TODO


class Cell(cellular.Cell):
    value = -1

    def setValue(self, int):
        value = int

class Player(cellular.Agent):

    def __init__(self):
        self.ai = None
        # alpha ... learning rate between 0-1 (0 means never update Q-values)
        # gamma ... discount factor between 0-1 (higher means the algorithm looks farther into the future - at 1 
        #           infinite rewards possible -> dont go to 1)
        # epsilon ... exploration factor between 0-1 (chance of taking a random action)
        #
        # set values, epsilon will be periodically overwritten (see pre train section farther down) until it reaches 0
        self.ai = learner.QLearn(actions=range(puzzleActionSize), alpha=0.1, gamma=0.9, epsilon=0.1)
        self.lastState = None
        self.lastAction = None
        self.solved = 0

    # calc reward based on current state (-1 default, +100 puzzle solved) and
    # if puzzle solved -> create random new puzzle
    # if this is not the first state after new puzzle created -> Q-learn(s,a,r,s')
    # choose an action and perform that action
    def update(self):
        # calculate the state of the surrounding cells (cat, cheese, wall, empty)
        puzzleState = self.world.getState()
        # assign a reward of -1 by default
        reward = -1

        # observe the reward and update the Q-value
        if self.world.isPuzzleSolved():
            self.solved += 1
            reward = 100
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, puzzleState)
            self.lastState = None

            createRandomPuzzle()
            return

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, puzzleState)

        # get updated state (puzzle might have been recreated after being solved), choose a new action and execute it
        puzzleState = self.world.getState()
        action = self.ai.chooseAction(puzzleState)
        self.lastState = puzzleState
        self.lastAction = action

        # move chosen tile, if it can not be moved do nothing
        self.moveTile(action)


# ----------------------------------
# start learning
# ----------------------------------

player = Player()

world = cellular.World(Cell, puzzleSize=puzzleSize)
world.age = 0
world.addAgent(player)


# how many time steps to pre train
steps = 10000

epsilonX = (0, steps*0.7)  # for how many time steps epsilon will be > 0, value experimental
epsilonY = (0.1, 0)  # TODO why does the mouse still learn with an exploration factor of 0?
# decay rate for epsilon so it hits 0 after epsilonX[1] time steps
epsilonM = (epsilonY[1] - epsilonY[0]) / (epsilonX[1] - epsilonX[0])

endAge = world.age + steps

# pre train the player till endAge
while world.age < endAge:
    # calls update on player (do action and learn) and then updates score and redraws screen
    world.update()

    # every 100 time steps, decay epsilon
    if world.age % 100 == 0:
        # this gradually decreases epsilon from epsilony[0] to epsilony[1] over the course of epsilonx[0] to [1]
        # -> at epsilonx[1] epsilon will reach epsilony[1] and stay there
        player.ai.epsilon = (epsilonY[0] if world.age < epsilonX[0] else
                            epsilonY[1] if world.age > epsilonX[1] else
                            epsilonM * (world.age - epsilonX[0]) + epsilonY[0])
        # alternatively just multiply by some factor... harder to set right i guess
        # player.ai.epsilon *= 0.9995

    # every 10.000 steps show current averageStepsPerPuzzle and stuff and then reset stats to measure next 10.000 steps
    if world.age % 10000 == 0:
        averageStepsPerPuzzle = 10000 / player.solved
        print "Age: {:d}, e: {:0.3f}, Solved: {:d}, average steps per puzzle: {:f}%" \
            .format(world.age, player.ai.epsilon, player.solved, averageStepsPerPuzzle)
        player.solved = 0


# ----------------------------------
# show off
# ----------------------------------

# after pre training - show off the mouse ai (while still training, but slower because it has to render now)
# PAGEUP to render less and thus learn faster
# PAGEDOWN to reverse the effect
# SPACE to pause

world.display.activate(size=30)
world.display.delay = 1
while 1:
    world.update(player.solved)
