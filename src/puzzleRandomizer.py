import math
import random


# import solverBySomeGuy


class Randomizer():

    def __init__(self, puzzleSize):
        self.puzzleSize = puzzleSize
        self.boardParts = [[0 for i in range(puzzleSize)] for j in range(puzzleSize)]
        self.emptyLocX = None
        self.emptyLocY = None

    # creates new random puzzle with world puzzleSize
    def makeRandomPuzzle(self, nSolved):
        # initate random puzzle
        self.initTiles(nSolved)
        # get position of empty cell
        self.calcEmptyLoc()
        # see if puzzle is solvable
        if (not (self.isSolvable(self.puzzleSize, self.puzzleSize, self.emptyLocY + 1))):
            # print("wrong: ",self.boardParts)
            # print(self.emptyLocX, self.emptyLocY)
            # print(self.boardParts)
            if ((self.emptyLocY == 0) & (self.emptyLocX <= 1)):
                self.swapTiles(self.puzzleSize - 2, self.puzzleSize - 1, self.puzzleSize - 1, self.puzzleSize - 1)
            else:
                self.swapTiles(0, 0, 1, 0)  # swap [0,0] [1,0], means swap position 0 and 1
            self.calcEmptyLoc()
            self.isSolvable(self.puzzleSize, self.puzzleSize, self.emptyLocY + 1)
            # print(self.boardParts)
        # print("fixed if broken: ", self.boardParts)

        # TODO temp for bugfixing
        # check if solvable
        # boardText = ""
        # for y in range(0, self.puzzleSize):
        #     for x in range(0, self.puzzleSize):
        #         boardText += str(self.boardParts[y][x]) + ","
        # boardText = boardText[:-1]
        # print(self.boardParts)
        # print(boardText)
        # b = solverBySomeGuy.Board(3, boardText)
        # b.get_solution()

        return self.boardParts
        # initEmpty()

    def calcEmptyLoc(self):
        for x in range(0, self.puzzleSize):
            for y in range(0, self.puzzleSize):
                if (self.boardParts[y][x] == 0):
                    self.emptyLocX = x
                    self.emptyLocY = y
                    return

    # creates puzzle in solved position, eg for size = 3
    # 1 2 3
    # 4 5 6
    # 7 8 0
    def createSolvedPosition(self):
        num = 1
        for y in range(0, self.puzzleSize):
            for x in range(0, self.puzzleSize):
                self.boardParts[y][x] = num
                num += 1
        self.boardParts[self.puzzleSize - 1][self.puzzleSize - 1] = 0

    def getRandomAdjacent(self, x, y):
        foundX = None
        foundY = None

        while foundX is None:
            dir = random.randint(1, 4)
            if dir == 1:
                if y > 0:
                    foundX = x
                    foundY = y - 1
            elif dir == 2:
                if x < self.puzzleSize - 1:
                    foundX = x + 1
                    foundY = y
            elif dir == 3:
                if y < self.puzzleSize - 1:
                    foundX = x
                    foundY = y + 1
            elif dir == 4:
                if x > 0:
                    foundX = x - 1
                    foundY = y
        return foundX, foundY

    # initiates Puzzle in completely random position using the Fisher-Yates algorithm
    def initTiles(self, nSolved):
        # create puzzle in solved position
        self.createSolvedPosition()
        # randomise puzzle
        if nSolved < 20:
            # make nSolved + 1 random moves to give easy puzzles at the beginning
            for i in range(nSolved + 1):
                self.calcEmptyLoc()
                randX, randY = self.getRandomAdjacent(self.emptyLocX, self.emptyLocY)
                self.swapTiles(self.emptyLocX, self.emptyLocY, randX, randY)
        else:
            i = self.puzzleSize * self.puzzleSize - 1
            while (i > 0):
                j = math.floor(random.random() * i)
                xi = int(i % self.puzzleSize)
                yi = int(math.floor(i / self.puzzleSize))
                xj = int(j % self.puzzleSize)
                yj = int(math.floor(j / self.puzzleSize))
                self.swapTiles(xi, yi, xj, yj)
                i -= 1

    def swapTiles(self, x1, y1, x2, y2):
        temp = self.boardParts[y1][x1]
        # print("1: %d" %temp)
        self.boardParts[y1][x1] = self.boardParts[y2][x2]
        # print("2: %d" %self.boardParts[y2][x2])
        self.boardParts[y2][x2] = temp

    def countInversions(self, y, x):
        inversions = 0

        position = y * self.puzzleSize + x
        nextPosition = position + 1
        lastPosition = self.puzzleSize ** 2 - 1

        tileValue = self.boardParts[y][x]

        while (nextPosition <= lastPosition):
            xNext = nextPosition % self.puzzleSize
            yNext = int(math.floor(nextPosition / self.puzzleSize))
            valueAtNextPosition = self.boardParts[yNext][xNext]

            if ((tileValue > valueAtNextPosition) & (valueAtNextPosition != 0)):
                inversions += 1

            nextPosition += 1

        return inversions

    def sumInversions(self):
        inversions = 0
        for y in range(0, self.puzzleSize):
            for x in range(0, self.puzzleSize):
                inversions += self.countInversions(y, x)
        return inversions

    def isSolvable(self, width, height, emptyRow):
        inversions = self.sumInversions()
        # print(inversions)

        # odd width
        if (width % 2 == 1):
            # print(self.sumInversions())
            # print("is solvable A: %s" %(inversions % 2 == 0))
            return (inversions % 2 == 0)

        # even width
        else:
            # empty on even row counted from the bottom
            if ((height - emptyRow) % 2 == 1):
                # print("is solvable B: %s" % (inversions % 2 == 1))
                return (inversions % 2 == 1)
            # empty on odd row counted from the bottom
            else:
                # print("is solvable C: %s" % (inversions % 2 == 0))
                return (inversions % 2 == 0)
