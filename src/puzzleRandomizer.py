import math
import random
import solverBySomeGuy


class Randomizer():

    def __init__(self, puzzleSize):
        self.puzzleSize = puzzleSize
        self.tileCount = puzzleSize #** 2
        self.boardParts = [[0 for i in range(puzzleSize)] for j in range(puzzleSize)]
        self.emptyLocX = None
        self.emptyLocY = None

    # creates new random puzzle with world puzzleSize
    def makeRandomPuzzle(self):
        # initate random puzzle
        self.initTiles()
        # get position of empty cell
        self.calcEmptyLoc()
        # see if puzzle is solvable
        if (not(self.isSolvable(self.tileCount, self.tileCount, self.emptyLocY + 1))):
            #print("wrong: ",self.boardParts)
            #print(self.emptyLocX, self.emptyLocY)
            print(self.boardParts)
            if ((self.emptyLocY == 0) & (self.emptyLocX <= 1)):
                self.swapTiles(self.tileCount - 2, self.tileCount - 1, self.tileCount - 1, self.tileCount - 1)
            else:
                self.swapTiles(0, 0, 1, 0) # swap [0,0] [1,0], means swap position 0 and 1
            self.calcEmptyLoc()
            self.isSolvable(self.tileCount, self.tileCount, self.emptyLocY + 1)
            print(self.boardParts)
        #print("fixed if broken: ", self.boardParts)


        # TODO temp for bugfixing
        # check if solvable
        boardText = ""
        for y in range(0, self.puzzleSize):
            for x in range(0, self.puzzleSize):
                boardText += str(self.boardParts[y][x]) + ","
        boardText = boardText[:-1]
        #print(self.boardParts)
        print(boardText)
        b = solverBySomeGuy.Board(3, boardText)
        b.get_solution()


        return self.boardParts
        #initEmpty()

    def calcEmptyLoc(self):
        for x in range(0, self.puzzleSize):
            for y in range(0, self.puzzleSize):
                if(self.boardParts[y][x] == 0):
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
        self.boardParts[self.puzzleSize-1][self.puzzleSize-1] = 0


    # initiates Puzzle in completely random position using the Fisher-Yates algorithm
    def initTiles(self):
        # create puzzle in solved position
        self.createSolvedPosition()
        # randomise puzzle
        i = self.tileCount * self.tileCount - 1
        while (i > 0):
            j = math.floor(random.random() * i)
            xi = int(i % self.tileCount)
            yi = int(math.floor(i / self.tileCount))
            xj = int(j % self.tileCount)
            yj = int(math.floor(j / self.tileCount))
            self.swapTiles(xi, yi, xj, yj)
            i -= 1

    def swapTiles(self, x1, y1, x2, y2):
        temp = self.boardParts[y1][x1]
        #print("1: %d" %temp)
        self.boardParts[y1][x1] = self.boardParts[y2][x2]
        #print("2: %d" %self.boardParts[y2][x2])
        self.boardParts[y2][x2] = temp

    def countInversions(self, i, j):
        inversions = 0
        position = j * self.tileCount + i
        lastTile = self.tileCount ** 2
        # tileValue = self.boardParts[i][j].y * self.tileCount + self.boardParts[i][j].x
        # TODO changed i and j -- should be right with j,i
        tileValue = self.boardParts[j][i]

        #print("-.-.---.-.-.-")
        #print(self.boardParts)
        #print(position)
        #print(tileValue)
        #print("--------")


        nextPosition = position + 1
        while (nextPosition < lastTile):
            k = nextPosition % self.tileCount
            l = int(math.floor(nextPosition / self.tileCount))
            # valueAtNextPosition = self.boardParts[k][l].y * self.tileCount + self.boardParts[k][l].x
            # TODO changed k and l, should be right with l,k
            valueAtNextPosition = self.boardParts[l][k]

            #print(self.boardParts)
            #print(nextPosition)
            #print(valueAtNextPosition)
            #print("++++++++++")

            # TODO was: lastTile - 1 .... if with -1 -> 8 is ignored -> swap with 8 doesnt change calc -> wrong results!
            # TODO if without -1 -> always gives true as result ????
            if ((tileValue > valueAtNextPosition) & (tileValue != (lastTile - 0))):
                inversions += 1
            nextPosition += 1

            #print(inversions)
        return inversions

    def sumInversions(self):
        inversions = 0
        for j in range(0, self.tileCount):
            for i in range(0, self.tileCount):
                inversions += self.countInversions(i, j)
                #i += 1
            #j += 1
        return inversions

    def isSolvable(self, width, height, emptyRow):
        if (width % 2 == 1):
            #print(self.sumInversions())
            print("is solvable A: %s" %(self.sumInversions() % 2 == 0))
            return (self.sumInversions() % 2 == 0)
        else:
            print("is solvable B: %s" % (self.sumInversions() + height - emptyRow) % 2 == 0)
            return ((self.sumInversions() + height - emptyRow) % 2 == 0)