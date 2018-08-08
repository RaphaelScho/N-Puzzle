import sys
import time


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
                        print('Error: invalid colour:', c)
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


# creates Display instance
def makeDisplay(world):
    d = Display()
    d.world = world
    return d


# creates title
def makeTitle(world):
    text = 'age: %d' % world.age
    extra = []
    if world.solved:
        extra.append('solved=%d' % world.solved)
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
