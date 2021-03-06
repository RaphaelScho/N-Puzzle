import math
import random

import puzzleRandomizer
from datetime import datetime
from copy import deepcopy
import itertools
import numpy as np


# ------------------------------------------------------------ #
# ------------------ SET PUZZLE SIZE HERE -------------------- #


algorithm = 1       # use  dictionary (0), neural network (1) or lgbm regressor (2)
puzzleSize = 3      # Size 3 means 3x3 puzzle


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #

print("initializing puzzle..")

if algorithm == 0:
    import qlearn as learner
elif algorithm == 1:
    import qlearn_nn as learner
elif algorithm == 2:
    import qlearn_lgbm as learner

if algorithm == 0:
    if puzzleSize == 2: # hardest instance takes 5 moves
        epsilonSteps = 1000   # over how many steps epsilon is reduced to its final value
        epsilonStartVal = 0.05   # chance to take a random action
        epsilonEndVal = 0.001
        alphaVal = 1          # learning rate
        gammaVal = 0.99          # discount factor for future rewards
        rewardVal = 1           # reward for solving the puzzle
        punishVal = -1      # punishment for doing nothing

    elif puzzleSize == 3: # hardest instance takes 31 moves to solve
        epsilonSteps = 4500000
        epsilonStartVal = 0.20
        epsilonEndVal = 0.01
        alphaVal = 1
        gammaVal = 0.999
        rewardVal = 1
        punishVal = -1
elif algorithm == 1:
    if puzzleSize == 2:
        epsilonSteps = 50000
        epsilonStartVal = 0.20
        epsilonEndVal = 0.01
        alphaVal = 0.01
        gammaVal = 0.99
        rewardVal = 1
        punishVal = -1

    elif puzzleSize == 3:
        epsilonSteps = 24500000
        epsilonStartVal = 0.9
        epsilonEndVal = 0.01
        alphaVal = 0.0001
        gammaVal = 0.95
        rewardVal = 10
        punishVal = -1
elif algorithm == 2:
    if puzzleSize == 2:
        epsilonSteps = None
        epsilonStartVal = 0.3
        epsilonEndVal = None
        alphaVal = 0.05
        gammaVal = 0.99
        rewardVal = 1
        punishVal = -1

    elif puzzleSize == 3:
        epsilonSteps = None
        epsilonStartVal = 1
        epsilonEndVal = None
        alphaVal = 0.05
        gammaVal = 0.99
        rewardVal = 100
        punishVal = -1


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #


class Puzzle():
    def __init__(self, puzzleSize):
        # alpha ... learning rate between 0-1 (0 means never update Q-values)
        # gamma ... discount factor between 0-1 (higher means the algorithm looks farther into the future - at 1 
        #           infinite rewards possible -> dont go to 1)
        # epsilon ... exploration factor between 0-1 (chance of taking a random action)

        # set values, epsilon will be periodically overwritten (see pre train section farther down) until it reaches 0
        # testing alpha = 1 instead of 0.1
        self.ai = learner.QLearn(puzzleSize = puzzleSize, epsilon=epsilonStartVal, alpha=alphaVal, gamma=gammaVal)
        self.lastState = None
        self.lastAction = None
        self.solved = 0
        self.age = 0
        # all tile swaps that have been done
        self.movesDone = 0
        # all actions that have been taken = all attempted swaps
        self.actionsTaken = 0
        self.puzzleSize = puzzleSize
        # 2d array containing values describing which numbers are in which positions [pos] = value
        # for size = 2, state[1][0] = c:
        # a b
        # c d
        self.randomizer = puzzleRandomizer.Randomizer(self.puzzleSize)
        # create random solvable puzzle start
        self.state = self.randomizer.makeRandomPuzzle(self.solved)
        # describes position of the empty cell (value = 0) (x,y)
        self.emptyCellPos = self.initEmptyCellPos()
        # up, down, left, right
        self.direction_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # create dict of cells in the puzzle that are neighbours to each other
        self.neighbours = self.initNeighbours()
        # create dict to get 2d-positions from 1d-position: (x,y)
        self.positionConverter = self.init1dTo2dPositionConverter()
        # create array equal to state, but with the expected solutions instead
        self.solution = self.initSolvedPosition()
        # self.display = display.makeDisplay(self)
        # init variables to calc averages
        self.solveCount = 0
        self.totalMoves = 0
        self.totalTime = 0
        self.steps = 0

        #self.currentManhattan = self.getManhattanDistance(self.state, self.solution)
        #self.lastManhattan = self.currentManhattan
        self.goalPositions = self.createGoalPositionsPerTile()

        # get manhattan distance for tile num at pos y,x via self.manhattanPerTile[num][(y,x)]
        self.manhattanPerTile = self.createManhattanPerTile()

        # get manhattan distance for a board state [[1,2,3],[4,5,6],[7,8,0]]
        # via self.manhattanPerBoard[(1,2,3,4,5,6,7,8,0)]
        self.manhattanPerBoard = self.createManhattanPerBoard()

    # create neighbours dict which has a list of neighbour-positions for each position
    def initNeighbours(self):
        neighbours = {}
        pos = 0

        for y in range(self.puzzleSize):
            for x in range(self.puzzleSize):
                n = []
                # space to the left
                if (x - 1 >= 0):
                    n.append([x - 1, y])
                # space to the right
                if (x + 1 < self.puzzleSize):
                    n.append([x + 1, y])
                # space above
                if (y - 1 >= 0):
                    n.append([x, y - 1])
                # space below
                if (y + 1 < self.puzzleSize):
                    n.append([x, y + 1])
                neighbours[pos] = n
                pos += 1

        return neighbours

    def init1dTo2dPositionConverter(self):
        conv = []
        for y in range(0, self.puzzleSize):
            for x in range(0, self.puzzleSize):
                conv.append(np.array([x, y]))
        return conv

    def initSolvedPosition(self):
        sol = [[0 for i in range(self.puzzleSize)] for j in range(self.puzzleSize)]
        num = 1
        for y in range(0, self.puzzleSize):
            for x in range(0, self.puzzleSize):
                sol[y][x] = num
                num += 1
        sol[self.puzzleSize - 1][self.puzzleSize - 1] = 0
        return sol

    def initEmptyCellPos(self):
        for x in range(0, self.puzzleSize):
            for y in range(0, self.puzzleSize):
                if (self.state[y][x] == 0):
                    return np.array([x, y])

    # try to move the tile at position
    # if that move is possible, move and return True
    # else do not move and return False
    def moveTile(self, direction):

        self.age += 1
        self.actionsTaken += 1

        # if the cell at position has the empty cell as a neighbour -> swap it with the empty cell and return True
        #if self.emptyCellPos in self.neighbours[position]:
        # print(self.emptyCellPos)
        # print(self.direction_list[direction])
        newPos = self.emptyCellPos + self.direction_list[direction]
        if not (np.sum(newPos >= self.puzzleSize) or np.sum(newPos < 0)):
            # curPos = self.positionConverter[direction]
            newPosX = newPos[0]
            newPosY = newPos[1]
            newPosValue = self.state[newPosY][newPosX]

            emptyCellPos = self.emptyCellPos
            emptyPosX = emptyCellPos[0]
            emptyPosY = emptyCellPos[1]

            # swap values in self.state
            self.state[emptyPosY][emptyPosX] = newPosValue
            self.state[newPosY][newPosX] = 0

            # set new emptyCellPos
            self.emptyCellPos = newPos

            self.movesDone += 1
            return True

        # else do not move and return False
        else:
            return False

    # calc reward based on current state and if puzzle solved -> create random new puzzle
    # if this is not the first state after new puzzle created -> Q-learn(s,a,r,s')
    # choose an action and perform that action
    def update(self):
        reward = None
        hasMoved = True
        currentState = deepcopy(self.state)

        if not self.isPuzzleSolved():
            # reward based on manhattan distance
            mh = self.getManhattanForBoard(currentState)
            #print("Manhattan: " + str(mh))
            if self.puzzleSize == 2:
                reward = ((1 / math.sqrt(mh) - 1) - 0.001) / 20000
            elif self.puzzleSize == 3:
                reward = ((1 / math.sqrt(mh) - 1) - 0.001) / 20

            # if last action was not legal -> useless action -> punish
            if (self.lastState == currentState):
                reward = punishVal
                hasMoved = False

        # observe the reward and update the Q-value
        else:
            endTime = datetime.now()

            reward = rewardVal
            if self.lastState is not None:
                if algorithm == 2:
                    self.ai.learn(self.lastState, self.lastAction, reward, currentState, True, hasMoved)
                else: #TODO why None instead of solved state?
                    self.ai.learn(self.lastState, self.lastAction, reward, None, True, hasMoved)
            self.lastState = None

            self.solved += 1
            self.solveCount += 1

            self.state = self.randomizer.makeRandomPuzzle(self.solved)
            self.emptyCellPos = self.initEmptyCellPos()

            self.steps = 0

            totalTime = endTime - self.startTime

            # calculate time difference
            timeDif = totalTime.seconds + 1.0 * totalTime.microseconds / 1000 / 1000

            # reset average calculation every few puzzles
            if self.solved % 35 == 0:
                self.totalTime = 0
                self.totalMoves = 0
                self.solveCount = 1
                print("\nresetting calculation of average")

            self.totalTime += timeDif
            self.totalMoves += self.movesDone

            if algorithm != 2:
                print((
                                  "\navg moves: %f \tavg time: %f seconds \tmoves: %d \ttime: %f seconds \tactions: %d \t\tepsilon: %f \tsolved: %d"
                                  % (self.totalMoves / (self.solveCount * 1.0), self.totalTime / (self.solveCount * 1.0),
                                     self.movesDone, timeDif, self.actionsTaken, self.ai.epsilon, self.solved)).expandtabs(
                    18))
                print(datetime.now())
                file.write(("%f,%f,%d,%f,%d,%f,%d\n"
                            % (self.totalMoves / (self.solveCount * 1.0), self.totalTime / (self.solveCount * 1.0),
                               self.movesDone, timeDif, self.actionsTaken, self.ai.epsilon, self.solved)).expandtabs(18))
                file.flush()
            else:
                print((
                        "\navg moves: %f \tavg time: %f seconds \tmoves: %d \ttime: %f seconds \tactions: %d \t\tepsilon: %f \tsolved: %d"
                        % (self.totalMoves / (self.solveCount * 1.0), self.totalTime / (self.solveCount * 1.0),
                           self.movesDone, timeDif, self.actionsTaken, self.ai.get_exploration_rate(), self.solved)).expandtabs(
                    18))
                print(datetime.now())
                file.write(("%f,%f,%d,%f,%d, %f, %d\n"
                            % (self.totalMoves / (self.solveCount * 1.0), self.totalTime / (self.solveCount * 1.0),
                               self.movesDone, timeDif, self.actionsTaken, self.ai.get_exploration_rate(), self.solved)).expandtabs(
                    18))
                file.flush()

            self.movesDone = 0
            self.actionsTaken = 0


            self.startTime = datetime.now()

            return

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, currentState, False, hasMoved)


        # get updated state (puzzle might have been recreated after being solved), choose a new action and execute it
        currentState = self.state
        action = self.ai.chooseAction(currentState)

        self.lastState = deepcopy(currentState)
        self.lastAction = action

        # move chosen tile, if it can not be moved do nothing
        self.moveTile(action)
        self.steps += 1

        if self.steps > 10000:
            print("resetting..")
            self.lastState = None
            self.state = self.randomizer.makeRandomPuzzle(self.solved)
            self.emptyCellPos = self.initEmptyCellPos()
            self.steps = 0

            #endTime = datetime.now()
            #totalTime = endTime - self.startTime
            # calculate time difference
            #timeDif = totalTime.seconds + 1.0 * totalTime.microseconds / 1000 / 1000
            #self.totalTime += timeDif
            #self.totalMoves += self.movesDone
            #self.movesDone = 0
            #self.actionsTaken = 0
            #self.startTime = datetime.now()

            if algorithm==2:
                self.ai.lgbm.last_memory.clear()
            elif algorithm==1:
                if random.random() > 0.1:
                    self.ai.batch.clear()
                    self.ai.mem_count = 0

            return

    def isPuzzleSolved(self):
        return (self.state == self.solution)

    def createGoalPositionsPerTile(self):
        goalPositions = {}
        for y in range(0, self.puzzleSize):
            for x in range(0, self.puzzleSize):
                goalPositions[self.solution[y][x]] = (y,x)
        return goalPositions

    def createManhattanPerTile(self):
        manhattanPerTile = {}
        # for each numbered tile calculate each position's manhattan distance
        for num in range(1,self.puzzleSize**2):
            manhattanForThisTile = {}
            for y in range(0, self.puzzleSize):
                for x in range(0, self.puzzleSize):
                    dif = abs(y - self.goalPositions[num][0]) + abs(x - self.goalPositions[num][1])
                    manhattanForThisTile[(y,x)] = dif
            manhattanPerTile[num] = manhattanForThisTile
        return manhattanPerTile

    def createManhattanPerBoard(self):
        manhattanPerBoard = {}
        numbers = range(0, self.puzzleSize**2)
        permutations = list(itertools.permutations(numbers))
        for perm in permutations:
            dist = 0
            # get manhattan distance for tile num at pos y,x via self.manhattanPerTile[num][(y,x)]
            for i in range(0,len(perm)):
                (x,y) = self.positionConverter[i]
                num = perm[i]
                if num != 0:
                    dist += self.manhattanPerTile[num][y,x]
            manhattanPerBoard[perm] = dist
        return manhattanPerBoard

    def getManhattanForBoard(self, state):
        # state like [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        flat_list = [item for sublist in state for item in sublist]
        dist = self.manhattanPerBoard[tuple(flat_list)]
        return dist


# ----------------------------------
# start learning
# ----------------------------------

puzzle = Puzzle(puzzleSize=puzzleSize)

if algorithm != 2:
    # learning factor
    epsilonX = (0, epsilonSteps)  # for how many time steps epsilon will be > 0, TODO value experimental
    epsilonY = (puzzle.ai.epsilon, epsilonEndVal) # start and end epsilon value
    # decay rate for epsilon so it hits the minimum value after epsilonX[1] time steps
    epsilonM = (epsilonY[1] - epsilonY[0]) / (epsilonX[1] - epsilonX[0])

puzzle.startTime = datetime.now()
print("puzzle start: %s" %puzzle.startTime)

# create log file

fname = ""
if algorithm == 1:
    fname = fname + "nn"
elif algorithm == 0:
    fname = fname + "simple"
elif algorithm == 2:
    fname = fname + "lgbm"
fname = "..\\log\\" + str(puzzleSize) + "\\" + fname + "_" + str(puzzle.startTime).replace(":", "-") + ".csv"
with open(fname,"w+") as file:
    print("Starting training..")
#with open("fname.name as.csv","w+") as file:
    #write header
    file.write("avg moves, avg time, moves, time, actions, epsilon, solved\n")

    # train the player
    #while puzzle.age < learningSteps:
    firstVictoryAge = 0
    while True:
        # calls update on puzzle (do action and learn) and then updates score and redraws screen
        puzzle.update()

        if algorithm != 2:
            # every 100 time steps, decay epsilon (only after first puzzle is solved)
            if (puzzle.solved > 0) & (puzzle.age % 100 == 0):
                relevantAge = puzzle.age - firstVictoryAge
                # this gradually decreases epsilon from epsilonY[0] to epsilonY[1] over the course of epsilonX[0] to [1]
                # -> at epsilonX[1] epsilon will reach epsilonY[1] and stay there
                puzzle.ai.epsilon = (epsilonY[0] if relevantAge < epsilonX[0] else
                                     epsilonY[1] if relevantAge > epsilonX[1] else
                                     epsilonM * (relevantAge - epsilonX[0]) + epsilonY[0])
                # alternatively just multiply by some factor... harder to set right I guess
                # puzzle.ai.epsilon *= 0.9995
            elif puzzle.solved < 0:
                firstVictoryAge = puzzle.age + 1

        # every .. steps show current averageStepsPerPuzzle and stuff and then reset stats to measure next ... steps
        if puzzle.age % 100000 == 0:
            print("\nage: " + str(puzzle.age))
            if algorithm != 2:
                print("epsilon: " + str(puzzle.ai.epsilon))
            else:
                print("epsilon: " + str(puzzle.ai.get_exploration_rate()))
            print(datetime.now())
            #print("manhattan: " + str(puzzle.getManhattanDistance(puzzle.state, puzzle.solution)))

            # print puzzle dict (for qlearn.py)
            #if(len(puzzle.ai.q) > 2200000):
            #    print("WRITING")
            #    f = open("output.txt","w")
            #    #f.write(str(puzzle.ai.q))
            #    for (key,value) in puzzle.ai.q.items():
            #        f.write("%s: %d%s" %(str(key), value,"\n"))
            #    f.close()
            #    break


    # ----------------------------------
    # show off
    # ----------------------------------

    endTime = datetime.now()
    totalTime = endTime - startTime
    print("total time: ", divmod(totalTime.days * 86400 + totalTime.seconds, 60))

    # after pre training - show off the player ai (while still training, but slower because it has to render now)
    # PAGEUP to render less and thus learn faster
    # PAGEDOWN to reverse the effect
    # SPACE to pause

    #puzzle.display.activate(size=30)
    #puzzle.display.delay = 1
    #print("enter show off mode")
    while 1 & False:
        puzzle.update()
