#import display
#import qlearn as learner
import qlearn_nn as learner
import puzzleRandomizer
from datetime import datetime

#import solverBySomeGuy

# ------------------------------------------------------------ #
# ------------------ SET PUZZLE SIZE HERE -------------------- #


puzzleSize = 3  # must be 3 or higher! Size 3 means 3x3 puzzle


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #


class Puzzle():
    def __init__(self, puzzleSize):
        # alpha ... learning rate between 0-1 (0 means never update Q-values)
        # gamma ... discount factor between 0-1 (higher means the algorithm looks farther into the future - at 1 
        #           infinite rewards possible -> dont go to 1)
        # epsilon ... exploration factor between 0-1 (chance of taking a random action)

        # set values, epsilon will be periodically overwritten (see pre train section farther down) until it reaches 0
        self.ai = learner.QLearn(puzzleSize = puzzleSize, alpha=0.1, gamma=0.95, epsilon=0.1)
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
        self.state = self.randomizer.makeRandomPuzzle()
        # describes position of the empty cell (value = 0) (x,y)
        self.emptyCellPos = self.initEmptyCellPos()
        # create dict of cells in the puzzle that are neighbours to each other
        self.neighbours = self.initNeighbours()
        # create dict to get 2d-positions from 1d-position: (x,y)
        self.positionConverter = self.init1dTo2dPositionConverter()
        # create array equal to state, but with the expected solutions instead
        self.solution = self.initSolvedPosition()
        # self.display = display.makeDisplay(self)
        # init variables to calc rolling averages
        self.rollLength = 10
        self.rollPos = 0
        self.timeList = []
        self.movesList = []

    # create neighbours dict which has a list of neighbour-positions for each position
    def initNeighbours(self):
        neighbours = {}
        pos = 0

        for y in range(self.puzzleSize):
            for x in range(self.puzzleSize):
                n = []
                # space to the left
                if (x - 1 >= 0):
                    n.append((x - 1, y))
                # space to the right
                if (x + 1 < self.puzzleSize):
                    n.append((x + 1, y))
                # space above
                if (y - 1 >= 0):
                    n.append((x, y - 1))
                # space below
                if (y + 1 < self.puzzleSize):
                    n.append((x, y + 1))
                neighbours[pos] = n
                pos += 1

        return neighbours

    def init1dTo2dPositionConverter(self):
        conv = []
        for y in range(0, self.puzzleSize):
            for x in range(0, self.puzzleSize):
                conv.append((x, y))
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
                    return (x, y)

    # try to move the tile at position
    # if that move is possible, move and return True
    # else do not move and return False
    def moveTile(self, position):

        self.age += 1
        self.actionsTaken += 1

        # if the cell at position has the empty cell as a neighbour -> swap it with the empty cell and return True
        if self.emptyCellPos in self.neighbours[position]:
            curPosTuple = self.positionConverter[position]
            curPosX = curPosTuple[0]
            curPosY = curPosTuple[1]
            curPosValue = self.state[curPosY][curPosX]

            emptyCellPosTuple = self.emptyCellPos
            empPosX = emptyCellPosTuple[0]
            empPosY = emptyCellPosTuple[1]

            # swap values in self.state
            self.state[empPosY][empPosX] = curPosValue
            self.state[curPosY][curPosX] = 0

            # set new emptyCellPos
            self.emptyCellPos = curPosTuple

            self.movesDone += 1
            return True

        # else do not move and return False
        else:
            #TODO the returns are not utilised
            return False

    # calc reward based on current state (-1 default, +100 puzzle solved) and
    # if puzzle solved -> create random new puzzle
    # if this is not the first state after new puzzle created -> Q-learn(s,a,r,s')
    # choose an action and perform that action
    def update(self):
        # self.display.update()
        # calculate the state of the surrounding cells (cat, cheese, wall, empty)
        currentState = self.state
        # assign a reward of -1 by default
        reward = -1

        # TODO maybe it is better to not selectively punish this
        # if last action was not legal -> useless action -> punish
        #if(self.lastState == currentState):
            #reward -= 2
            #pass

        # observe the reward and update the Q-value
        if self.isPuzzleSolved():
            self.solved += 1

            endTime = datetime.now()
            totalTime = endTime - self.startTime

            # calculate time difference
            timeDif = totalTime.seconds + 1.0 * totalTime.microseconds / 1000 / 1000

            # calculate rolling averages
            #if len(self.timeList)<10:
            self.timeList.append(timeDif)
            #else:
            #    self.timeList[self.rollPos] = timeDif
            #if len(self.movesList) < self.rollLength:
            self.movesList.append(self.movesDone)
            #else:
            #    self.movesList[self.rollPos] = self.movesDone

            #if self.rollPos >= (self.rollLength - 1):
            #    self.rollPos = 0
            #else:
            self.rollPos += 1
            # print rolling averages
            print(("avg moves: %d \tavg time: %f seconds \tmoves: %d \ttime: %f seconds \t\tepsilon: %f \tsolved: %f"
                  %(1.0*sum(self.movesList)/len(self.movesList), 1.0*sum(self.timeList)/len(self.timeList),
                    self.movesDone, timeDif, self.ai.epsilon, self.solved)).expandtabs(18))
            self.movesDone = 0
            self.actionsTaken = 0

            reward = 150
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, currentState)
            self.lastState = None

            self.state = self.randomizer.makeRandomPuzzle()
            self.emptyCellPos = self.initEmptyCellPos()

            self.startTime = datetime.now()

            return

        # MODIFICATION TEST: stop game after some amount of steps and start new puzzle
        # currently DEACTIVATED
        #if (False & self.actionsTaken >= 7000):
        #    # reward = -100
        #    self.ai.learn(self.lastState, self.lastAction, reward, currentState)
        #    self.lastState = None
        #    self.state = self.randomizer.makeRandomPuzzle()
        #    self.emptyCellPos = self.initEmptyCellPos()
        #    self.movesDone = 0
        #    self.actionsTaken = 0
            #print("Puzzle canceled")


        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, currentState)

        # get updated state (puzzle might have been recreated after being solved), choose a new action and execute it
        currentState = self.state
        action = self.ai.chooseAction(currentState)

        self.lastState = currentState
        self.lastAction = action

        # move chosen tile, if it can not be moved do nothing
        self.moveTile(action)

    def isPuzzleSolved(self):
        if (self.state == self.solution):
            return True
        else:
            return False

    #def getCellValue(self, x, y):
    #    return self.state[self.puzzleSize * y + x]

    #def getCellValueByIndex(self, position):
    #    return self.state[position]


# ----------------------------------
# start learning
# ----------------------------------

if(puzzleSize < 3):
    print("puzzleSize too small, set to 3 instead!")
    puzzleSize = 3

puzzle = Puzzle(puzzleSize=puzzleSize)

# how many time steps to pre train
learningSteps = 200000000


# TODO is the initially set epsilon value just overwritten immediately?
# learning factor
epsilonX = (0, learningSteps * 0.7)  # for how many time steps epsilon will be > 0, TODO value experimental
epsilonY = (0.15, 0)
# decay rate for epsilon so it hits 0 after epsilonX[1] time steps
epsilonM = (epsilonY[1] - epsilonY[0]) / (epsilonX[1] - epsilonX[0])

puzzle.startTime = datetime.now()
print("puzzle start: %s" %puzzle.startTime)

# pre train the player
#while puzzle.age < learningSteps:
while True:
#while len(puzzle.ai.q) < 3600000:
    # calls update on puzzle (do action and learn) and then updates score and redraws screen
    puzzle.update()

    # every 100 time steps, decay epsilon
    if (puzzle.age % 100 == 0):
        # this gradually decreases epsilon from epsilonY[0] to epsilonY[1] over the course of epsilonX[0] to [1]
        # -> at epsilonX[1] epsilon will reach epsilonY[1] and stay there
        puzzle.ai.epsilon = (epsilonY[0] if puzzle.age < epsilonX[0] else
                             epsilonY[1] if puzzle.age > epsilonX[1] else
                             epsilonM * (puzzle.age - epsilonX[0]) + epsilonY[0])
        # alternatively just multiply by some factor... harder to set right I guess
        # puzzle.ai.epsilon *= 0.9995

    # every 10.000 steps show current averageStepsPerPuzzle and stuff and then reset stats to measure next 10.000 steps
    if puzzle.age % 10000 == 0:
        print(puzzle.age)
        #if(puzzle.solved > 0):
        #    averageStepsPerPuzzle = puzzle.age / puzzle.solved
        #else:
        #    averageStepsPerPuzzle = 0

        #print("length of learner db: %d" % (len(puzzle.ai.q)))
        #print "Age: {:d}, e: {:0.3f}, Solved: {:d}, average steps per puzzle: {:f}" \
        #print "Age: {:d}, e: {:0.3f}, Solved: {:d}" \
        #    .format(puzzle.age, puzzle.ai.epsilon, puzzle.solved)#,averageStepsPerPuzzle)
        #print("Legal puzzle swap attempts: %f%% " %(puzzle.movesDone/float(puzzle.actionsTaken)*100))
        #print("total actions: %d" %(puzzle.actionsTaken))
        # puzzle.solved = 0

        '''
        if(len(puzzle.ai.q) > 2200000):
            print("WRITING")
            f = open("output.txt","w")
            #f.write(str(puzzle.ai.q))
            for (key,value) in puzzle.ai.q.items():
                f.write("%s: %d%s" %(str(key), value,"\n"))
            f.close()
            break
        '''

        #if puzzle.age % 1000000 == 0:
            #print(puzzle.ai.q)

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
