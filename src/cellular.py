import random
import sys
import time


class Cell:
    # returns list of cells around x,y position of the cell
    def __getattr__(self, key):
        # TODO is this calculated every time that cell.neighbour is called? if so why is it stored in __dict__ then?
        if key == "neighbour":
            pts = [self.world.getPointInDirection(self.x, self.y, dir) for dir in range(self.world.directions)]
            ns = tuple([self.world.grid[y][x] for (x,y) in pts])
            # TODO see above (does this even do anything?)
            #self.__dict__["neighbour"] = ns
            return ns
        raise AttributeError(key)


class Agent:
    def __setattr__(self, key, val):
        if key == 'cell':
            old = self.__dict__.get(key, None)
            if old is not None:
                old.agents.remove(self)
            if val is not None:
                val.agents.append(self)
        self.__dict__[key] = val

    # move to target if it is not a wall
    def moveTile(self, position):
        target = self.cell.neighbour[position]
        # if target is a wall -> do not move and return false
        if getattr(target, 'wall', False):
            # print "hit a wall"
            return False
        # else move to target and return true
        self.cell = target
        return True

    # only used by Cat
    # calculates manhattan distance to target (mouse) and takes step in direction of shortest distance if not a wall
    def goTowards(self, target):
        if self.cell == target:
            return
        best = None
        for n in self.cell.neighbour:
            if n == target:
                best = target
                break
            dist = (n.x - target.x) ** 2 + (n.y - target.y) ** 2
            if best is None or bestDist > dist:
                best = n
                bestDist = dist
        if best is not None:
            if getattr(best, 'wall', False):
                return
            self.cell = best


class World:

    state = []

    # create random puzzle on init
    def __init__(self, cell=None, puzzleSize=3):
        if cell is None:
            cell = Cell
        self.Cell = cell
        self.display = makeDisplay(self)
        self.puzzleSize = puzzleSize
        self.solved = None
        self.reset()

    def getState(self):
        return self.state

    def isPuzzleSolved(self):
        return False
        # TODO do calc based on self.state and puzzle size

    def getCell(self, x, y):
        return self.grid[y][x]

    def getWrappedCell(self, x, y):
        return self.grid[y % self.height][x % self.width]

    # creates new cells for each position in the world, empties list of agents and sets age to 0
    def reset(self):
        self.grid = [[self.makeCell(i, j) for i in range(self.width)] for j in range(self.height)]
        # self.dictBackup = [[{} for i in range(self.width)] for j in range(self.height)]
        self.agents = []
        self.age = 0

    # creates new cell at x,y position
    def makeCell(self, x, y):
        c = self.Cell()
        c.x = x
        c.y = y
        c.world = self
        c.agents = []
        return c

    # loads the world from the text file and sets cell wall attributes to True or False based on world txt
    def load(self, f):
        if not hasattr(self.Cell, 'setWallColour'):
            return
        if isinstance(f, type('')):
            f = file(f)
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        fh = len(lines)
        fw = max([len(x) for x in lines])
        if fh > self.height:
            fh = self.height
            starty = 0
        else:
            starty = (self.height - fh) / 2
        if fw > self.width:
            fw = self.width
            startx = 0
        else:
            startx = (self.width - fw) / 2

        self.reset()
        # goes through each row of the world txt file and calls setWallColour on the respective cell which just checks
        # whether the file contains an X at the location and sets its wall attribute to true if so
        for j in range(fh):
            line = lines[j]
            for i in range(min(fw, len(line))):
                self.grid[starty + j][startx + i].setWallColour(line[i])

    # calls update on cat, mouse (and cheese) (change position and learn) and then updates score and redraws screen
    def update(self, fed=None, eaten=None):
        for a in self.agents:
            oldCell = a.cell
            a.update()
            if oldCell != a.cell:
                self.display.redrawCell(oldCell.x, oldCell.y)
            self.display.redrawCell(a.cell.x, a.cell.y)
        # end else
        if (fed):
            self.fed = fed
        if (eaten):
            self.eaten = eaten
        self.display.update()
        self.age += 1

    # returns x,y coordinates when going in direction dir
    # wraps around borders
    def getPointInDirection(self, x, y, dir):
        # dir is number 0-7
        # -> choose element 0-7 from array [(0,-1),...] e.g.: dir=0 -> dx=0, dy=-1
        if self.directions == 8:
            dx, dy = [(0, -1), (1, -1), (
                1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]
        elif self.directions == 4:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]
        elif self.directions == 6:
            if y % 2 == 0:
                dx, dy = [(1, 0), (0, 1), (-1, 1), (-1, 0),
                          (-1, -1), (0, -1)][dir]
            else:
                dx, dy = [(1, 0), (1, 1), (0, 1), (-1, 0),
                          (0, -1), (1, -1)][dir]

        x2 = x + dx
        y2 = y + dy

        if x2 < 0:
            x2 += self.width
        if y2 < 0:
            y2 += self.height
        if x2 >= self.width:
            x2 -= self.width
        if y2 >= self.height:
            y2 -= self.height

        return (x2, y2)

    # adds an agent and gives them a position in the world
    def addAgent(self, agent, x=None, y=None, cell=None, dir=None):
        self.agents.append(agent)
        if cell is not None:
            x = cell.x
            y = cell.y
        if x is None:
            x = random.randrange(self.width)
        if y is None:
            y = random.randrange(self.height)
        if dir is None:
            dir = random.randrange(self.directions)
        agent.cell = self.grid[y][x]
        agent.dir = dir
        agent.world = self


# creates Display instance
def makeDisplay(world):
    d = Display()
    d.world = world
    return d

# default Display
class PygameDisplay:
    activated = False
    paused = False
    title = ''
    updateEvery = 1
    delay = 0
    screen = None

    def activate(self, size=4):
        self.size = size
        pygame.init()
        w = self.world.width * size
        h = self.world.height * size
        if self.world.directions == 6:
            w += size / 2
        if PygameDisplay.screen is None or PygameDisplay.screen.get_width() != w or PygameDisplay.screen.get_height() != h:
            PygameDisplay.screen = pygame.display.set_mode(
                (w, h), pygame.RESIZABLE, 32)
        self.activated = True
        self.defaultColour = self.getColour(self.world.grid[0][0].__class__())
        self.redraw()

    def redraw(self):
        if not self.activated:
            return
        self.screen.fill(self.defaultColour)
        hexgrid = self.world.directions == 6
        self.offsety = (
                           self.screen.get_height() - self.world.height * self.size) / 2
        self.offsetx = (
                           self.screen.get_width() - self.world.width * self.size) / 2
        sy = self.offsety
        odd = False
        for row in self.world.grid:
            sx = self.offsetx
            if hexgrid and odd:
                sx += self.size / 2
            for cell in row:
                if len(cell.agents) > 0:
                    c = self.getColour(cell.agents[0])
                else:
                    c = self.getColour(cell)
                if c != self.defaultColour:
                    try:
                        self.screen.fill(c, (sx, sy, self.size, self.size))
                    except TypeError:
                        print 'Error: invalid colour:', c
                sx += self.size
            odd = not odd
            sy += self.size

    def redrawCell(self, x, y):
        if not self.activated:
            return
        sx = x * self.size + self.offsetx
        sy = y * self.size + self.offsety
        if y % 2 == 1 and self.world.directions == 6:
            sx += self.size / 2

        cell = self.world.grid[y][x]
        if len(cell.agents) > 0:
            c = self.getColour(cell.agents[0])
        else:
            c = self.getColour(cell)

        self.screen.fill(c, (sx, sy, self.size, self.size))

    # manages user inputs, delay, skips and calls to redraw the screen
    def update(self):
        if not self.activated:
            return
        if self.world.age % self.updateEvery != 0 and not self.paused:
            return
        self.setTitle(self.title)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
            elif event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                self.onResize(event)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_PAGEUP:
                if self.delay > 0:
                    self.delay -= 1
                else:
                    self.updateEvery *= 2
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_PAGEDOWN:
                if self.updateEvery > 1:
                    self.updateEvery /= 2
                else:
                    self.delay += 1
                if self.delay > 10:
                    self.delay = 10
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.pause()

        pygame.display.flip()
        if self.delay > 0:
            time.sleep(self.delay * 0.1)

    def setTitle(self, title):
        if not self.activated:
            return
        self.title = title
        title += ' %s' % makeTitle(self.world)
        if pygame.display.get_caption()[0] != title:
            pygame.display.set_caption(title)

    def pause(self, event=None):
        self.paused = not self.paused
        while self.paused:
            self.update()

    def onResize(self, event):
        if not self.activated:
            return
        pygame.display.set_mode(event.size, pygame.RESIZABLE, 32)
        oldSize = self.size
        scalew = event.size[0] / self.world.width
        scaleh = event.size[1] / self.world.height
        self.size = min(scalew, scaleh)
        if self.size < 1:
            self.size = 1
        self.redraw()

    def getColour(self, obj):
        # c = obj.colour is basically the same thing but without the default value
        c = getattr(obj, 'colour', 'white')
        if callable(c):
            c = c()
        return pygame.color.Color(c)

    def saveImage(self, filename=None):
        if filename is None:
            filename = '%05d.bmp' % self.world.age
        pygame.image.save(self.screen, filename)

# creates title
def makeTitle(world):
    text = 'age: %d' % world.age
    extra = []
    if world.fed:
        extra.append('fed=%d' % world.fed)
    if world.eaten:
        extra.append('eaten=%d' % world.eaten)
    if world.display.paused:
        extra.append('paused')
    if world.display.updateEvery != 1:
        extra.append('skip=%d' % world.display.updateEvery)
    if world.display.delay > 0:
        extra.append('delay=%d' % world.display.delay)

    if len(extra) > 0:
        text += ' [%s]' % ', '.join(extra)
    return text



try:
    import pygame

    Display = PygameDisplay
except:
    print("ERROR: Failed creating PyGameDisplay!!")
