import numpy as np
import pygame
from time import sleep
from pygame.locals import *
import keras
import gym
"""
encoding:
nothing 0 #000000
I 1 #00ffff
O 2 #ffff00
T 3 #ff00ff
S 4 #00ff00
Z 5 #ff0000
L 6 #ff7f00
J 7 #0000ff

examples.R
000 empty space
021 player piece, I
013 board piece, T

"""






class Tetris(gym.Env):
    def init(self):
        self.board = np.zeros((16, 26), dtype=int)
        self.board[0] = [10] * 26
        self.board[13] = [10] * 26
        self.board[1] = [10] * 26
        self.board[14] = [10] * 26
        self.board[2] = [10] * 26
        self.board[15] = [10] * 26
        for i in range(3, 13):
            self.board[i, 23] = 10
            self.board[i, 24] = 10
            self.board[i, 25] = 10
        self.nextsequence = []
        self.nextsequence += (np.random.permutation(7) + 1).tolist()
        self.nextsequence += (np.random.permutation(7) + 1).tolist()
        self.currentpositions = [(10, 23), (10, 23), (10, 23), (10, 23)]
        self.holdstate = False
        self.holdpiece = 0
        self.harddropstate = False
        self.leftdas = 0
        self.rightdas = 0
        self.cwstate = False
        self.ccwstate = False
        self.rotationstate = 0 #cw rotation increases
        self.score = 0
        self.tickcount = 0
        self.reward = 0
        self.action_space = gym.spaces.Discrete(7)
        obs_size = [8] * 236 + [2] * 7
        self.observation_space = gym.spaces.MultiDiscrete(obs_size)
        self.lastclear = ""

    def init_playable(self):
        pygame.init()
        self.screen = pygame.display.set_mode((440, 480))
        pygame.display.set_caption('bruh')
        self.clock = pygame.time.Clock()
        self.empty_square   = pygame.Surface((20, 20))
        self.empty_square.convert()
        self.empty_square.fill((  0,  0,  0))
        self.cyan_square   = pygame.Surface((20, 20))
        self.cyan_square.convert()
        self.cyan_square.fill((  0, 255, 255))
        self.yellow_square = pygame.Surface((20, 20))
        self.yellow_square.convert()
        self.yellow_square.fill((255, 255,   0))
        self.purple_square = pygame.Surface((20, 20))
        self.purple_square.convert()
        self.purple_square.fill((255,   0, 255))
        self.green_square  = pygame.Surface((20, 20))
        self.green_square.convert()
        self.green_square.fill((  0, 255,   0))
        self.red_square    = pygame.Surface((20, 20))
        self.red_square.convert()
        self.red_square.fill((255,   0,   0))
        self.orange_square = pygame.Surface((20, 20))
        self.orange_square.convert()
        self.orange_square.fill((255, 127,   0))
        self.blue_square   = pygame.Surface((20, 20))
        self.blue_square.convert()
        self.blue_square.fill((  0,   0, 255))

        self.empty = pygame.Surface((80, 40))
        self.empty.convert()
        self.empty.fill((0, 0, 0))

        self.I = pygame.Surface((80, 40))
        self.I.convert()
        self.I.fill((0, 0, 0))
        self.I.blit(self.cyan_square, ( 0, 0))
        self.I.blit(self.cyan_square, (20, 0))
        self.I.blit(self.cyan_square, (40, 0))
        self.I.blit(self.cyan_square, (60, 0))

        self.O = pygame.Surface((80, 40))
        self.O.convert()
        self.O.fill((0, 0, 0))
        self.O.blit(self.yellow_square, (20,  0))
        self.O.blit(self.yellow_square, (40,  0))
        self.O.blit(self.yellow_square, (20, 20))
        self.O.blit(self.yellow_square, (40, 20))

        self.T = pygame.Surface((80, 40))
        self.T.convert()
        self.T.fill((0, 0, 0))
        self.T.blit(self.purple_square, (20,  0))
        self.T.blit(self.purple_square, ( 0, 20))
        self.T.blit(self.purple_square, (20, 20))
        self.T.blit(self.purple_square, (40, 20))

        self.S = pygame.Surface((80, 40))
        self.S.convert()
        self.S.fill((0, 0, 0))
        self.S.blit(self.green_square, (20,  0))
        self.S.blit(self.green_square, (40,  0))
        self.S.blit(self.green_square, ( 0, 20))
        self.S.blit(self.green_square, (20, 20))

        self.Z = pygame.Surface((80, 40))
        self.Z.convert()
        self.Z.fill((0, 0, 0))
        self.Z.blit(self.red_square, ( 0,  0))
        self.Z.blit(self.red_square, (20,  0))
        self.Z.blit(self.red_square, (20, 20))
        self.Z.blit(self.red_square, (40, 20))

        self.L = pygame.Surface((80, 40))
        self.L.convert()
        self.L.fill((0, 0, 0))
        self.L.blit(self.orange_square, (40,  0))
        self.L.blit(self.orange_square, ( 0, 20))
        self.L.blit(self.orange_square, (20, 20))
        self.L.blit(self.orange_square, (40, 20))

        self.J = pygame.Surface((80, 40))
        self.J.convert()
        self.J.fill((0, 0, 0))
        self.J.blit(self.blue_square, ( 0,  0))
        self.J.blit(self.blue_square, ( 0, 20))
        self.J.blit(self.blue_square, (20, 20))
        self.J.blit(self.blue_square, (40, 20))

        self.pieces = {
            0: self.empty,
            1: self.I,
            2: self.O,
            3: self.T,
            4: self.S,
            5: self.Z,
            6: self.L,
            7: self.J
        }

        self.squares = {
            0: self.empty_square,
            1: self.cyan_square,
            2: self.yellow_square,
            3: self.purple_square, 
            4: self.green_square,
            5: self.red_square, 
            6: self.orange_square, 
            7: self.blue_square
        }

        #self.font = pygame.font.SysFont("firacode", 20)

    def __init__(self, human_playable):
        super(Tetris, self).__init__()
        self.init()
        if human_playable:
            self.init_playable()

    spawnpositions = {
        1: [(6, 1), (7, 1), (8, 1), (9, 1)],
        2: [(7, 1), (8, 1), (7, 2), (8, 2)],
        3: [(6, 2), (7, 1), (8, 2), (7, 2)],
        4: [(6, 2), (7, 2), (7, 1), (8, 1)],
        5: [(6, 1), (7, 1), (7, 2), (8, 2)],
        6: [(8, 1), (6, 2), (7, 2), (8, 2)],
        7: [(6, 1), (6, 2), (7, 2), (8, 2)]
    }

    rotateoffsetscw = {
        1: [[[(+2, -1), (+1,  0), ( 0, +1), (-1, +2)], (-2,  0), (+3,  0), (-3, +1), (+3, -3)], #0-1
            [[(+1, +2), ( 0, +1), (-1,  0), (-2, -1)], (-1,  0), (+3,  0), (-3, -2), (+3, +3)], #1-2
            [[(-2, +1), (-1,  0), ( 0, -1), (+1, -2)], (+2,  0), (-3,  0), (+3, -1), (+3, +3)], #2-3
            [[(-1, -2), ( 0, -1), (+1,  0), (+2, +1)], (+1,  0), (-3,  0), (+3, +2), (-3, -3)]], #3-0
        2: [[(0, 0)], 
            [(0, 0)], 
            [(0, 0)], 
            [(0, 0)]],
        3: [[[(+1, -1), (+1, +1), (-1, +1), ( 0,  0)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(+1, +1), (-1, +1), (-1, -1), ( 0,  0)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(-1, +1), (-1, -1), (+1, -1), ( 0,  0)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(-1, -1), (+1, -1), (+1, +1), ( 0,  0)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        4: [[[(+1, -1), ( 0,  0), (+1, +1), ( 0, +2)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(+1, +1), ( 0,  0), (-1, +1), (-2,  0)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(-1, +1), ( 0,  0), (-1, -1), ( 0, -2)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(-1, -1), ( 0,  0), (+1, -1), (+2,  0)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        5: [[[(+2,  0), (+1, +1), ( 0,  0), (-1, +1)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[( 0, +2), (-1, +1), ( 0,  0), (-1, -1)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(-2,  0), (-1, -1), ( 0,  0), (+1, -1)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[( 0, -2), (+1, -1), ( 0,  0), (+1, +1)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        6: [[[(-1, +1), ( 0, -2), (-1, -1), (-2,  0)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(-1, -1), (+2,  0), (+1, -1), ( 0, -2)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(+1, -1), ( 0, +2), (+1, +1), (+2,  0)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(+1, +1), (-2,  0), (-1, +1), ( 0, +2)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        7: [[[(+1, -1), ( 0, -2), (-1, -1), (-2,  0)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(+1, +1), (+2,  0), (+1, -1), ( 0, -2)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(-1, +1), ( 0, +2), (+1, +1), (+2,  0)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(-1, -1), (-2,  0), (-1, +1), ( 0, +2)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
    }

    rotateoffsetsccw = {
        1: [[[(+1, +2), ( 0, +1), (-1,  0), (-2, -1)], (-1,  0), (+3,  0), (-3, -2), (+3, +3)], #0+3
            [[(-2, +1), (-1,  0), ( 0, -1), (+1, -2)], (+2,  0), (-3,  0), (+3, -1), (-3, +3)], #3+2
            [[(-1, -2), ( 0, -1), (+1,  0), (+2, +1)], (+1,  0), (-3,  0), (+3, +2), (-3, -3)], #2+1
            [[(+2, -1), (+1,  0), ( 0, +1), (-1, +2)], (-2,  0), (+3,  0), (-3, +1), (+3, -3)]], #1+0
        2: [[(0, 0)], 
            [(0, 0)], 
            [(0, 0)], 
            [(0, 0)]],
        3: [[[(+1, +1), (-1, +1), (-1, -1), ( 0,  0)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)],
            [[(-1, +1), (-1, -1), (+1, -1), ( 0,  0)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(-1, -1), (+1, -1), (+1, +1), ( 0,  0)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(+1, -1), (+1, +1), (-1, +1), ( 0,  0)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        4: [[[(+1, +1), ( 0,  0), (-1, +1), (-2,  0)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(-1, +1), ( 0,  0), (-1, -1), ( 0, -2)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(-1, -1), ( 0,  0), (+1, -1), (+2,  0)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(+1, -1), ( 0,  0), (+1, +1), ( 0, +2)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        5: [[[( 0, +2), (-1, +1), ( 0,  0), (-1, -1)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(-2,  0), (-1, -1), ( 0,  0), (+1, -1)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[( 0, -2), (+1, -1), ( 0,  0), (+1, +1)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(+2,  0), (+1, +1), ( 0,  0), (-1, +1)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        6: [[[(-1, -1), (+2,  0), (+1, -1), ( 0, -2)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(+1, -1), ( 0, +2), (+1, +1), (+2,  0)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(+1, +1), (-2,  0), (-1, +1), ( 0, +2)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(-1, +1), ( 0, -2), (-1, -1), (-2,  0)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
        7: [[[(+1, +1), (+2,  0), (+1, -1), ( 0, -2)], (+1,  0), ( 0, -1), (-1, +3), (+1,  0)], 
            [[(-1, +1), ( 0, +2), (+1, +1), (+2,  0)], (+1,  0), ( 0, +1), (-1, -3), (+1,  0)], 
            [[(-1, -1), (-2,  0), (-1, +1), ( 0, +2)], (-1,  0), ( 0, -1), (+1, +3), (-1,  0)], 
            [[(+1, -1), ( 0, -2), (-1, -1), (-2,  0)], (-1,  0), ( 0, +1), (+1, -3), (-1,  0)]],
    }

    def modifypositions(self, changes):
        return list(map(tuple, np.add(self.currentpositions, changes).tolist()))

    def rotatecw(self):
        oldpositions = self.currentpositions
        for position in self.currentpositions:
            self.board[position] = 0
        counter = 0
        success = False
        for shift in self.rotateoffsetscw[self.currentpiece][self.rotationstate]:
            self.currentpositions = self.modifypositions(shift)
            counter = 0
            for position in self.currentpositions:
                if self.board[position] > 0:
                    counter += 1
            if counter == 0:
                success = True
                break
        if success:
            self.rotationstate = (self.rotationstate + 1) % 4
        else:
            self.currentpositions = oldpositions
        for position in self.currentpositions:
            self.board[position] = self.currentpiece
    
    def rotateccw(self):
        oldpositions = self.currentpositions
        for position in self.currentpositions:
            self.board[position] = 0
        counter = 0
        success = False
        for shift in self.rotateoffsetsccw[self.currentpiece][self.rotationstate]:
            self.currentpositions = self.modifypositions(shift)
            counter = 0
            for position in self.currentpositions:
                if self.board[position] > 0:
                    counter += 1
            if counter == 0:
                success = True
                break
        if success:
            self.rotationstate = (self.rotationstate - 1) % 4
        else:
            self.currentpositions = oldpositions
        for position in self.currentpositions:
            self.board[position] = self.currentpiece
    
    def advancepiece(self):
        #for position in self.currentpositions:
        #    self.board[position] += 10
        self.currentpiece = self.nextsequence.pop(0)
        self.rotationstate = 0
        self.currentpositions = self.spawnpositions[self.currentpiece]
        for position in self.currentpositions:
            self.board[position] = self.currentpiece
        if len(self.nextsequence) < 7:
            self.nextsequence += (np.random.permutation(7) + 1).tolist()

    def draw_board(self):
        background = pygame.Surface(self.screen.get_size())
        background.convert()
        background.fill((0, 0, 0))

        try:
            background.blit(self.pieces[self.holdpiece], (20, 20))
        except:
            background.blit(self.pieces[0], (20, 20))
        background.blit(self.pieces[self.nextsequence[0]], (340,  20))
        background.blit(self.pieces[self.nextsequence[1]], (340,  80))
        background.blit(self.pieces[self.nextsequence[2]], (340, 140))
        background.blit(self.pieces[self.nextsequence[3]], (340, 200))
        background.blit(self.pieces[self.nextsequence[4]], (340, 260))
        background.blit(self.pieces[self.nextsequence[5]], (340, 320))

        for x in range(3, 13):
            for y in range(1, 23):
                background.blit(self.squares[self.board[x, y] % 10], (60 + 20 * x, 20 + 20 * y))

        #text = self.font.render(str(self.score), True, (0, 0, 0))
        #background.blit(text, (0, 600))

        self.screen.blit(background, (0, 0))
        pygame.display.flip()

    def softdrop(self):
        nextpositions = self.modifypositions((0, 1))
        for position in self.currentpositions:
            self.board[position] = 0
        for position in nextpositions:
            if self.board[position] > 0:
                for position in self.currentpositions:
                    self.board[position] = self.currentpiece
                return
        self.currentpositions = nextpositions
        for position in self.currentpositions:
            self.board[position] = self.currentpiece

    def leftmove(self):
        nextpositions = self.modifypositions((-1, 0))
        for position in self.currentpositions:
            self.board[position] = 0
        for position in nextpositions:
            if self.board[position] > 0:
                for position in self.currentpositions:
                    self.board[position] = self.currentpiece
                return
        self.currentpositions = nextpositions
        for position in self.currentpositions:
            self.board[position] = self.currentpiece

    def rightmove(self):
        nextpositions = self.modifypositions((1, 0))
        for position in self.currentpositions:
            self.board[position] = 0
        for position in nextpositions:
            if self.board[position] > 0:
                for position in self.currentpositions:
                    self.board[position] = self.currentpiece
                return
        self.currentpositions = nextpositions
        for position in self.currentpositions:
            self.board[position] = self.currentpiece

    def harddrop(self):
        for _ in range(21):
            try:
                self.softdrop()
            except:
                print(self.currentpositions)
                print(self.board)
                print(self.currentpiece)

    def update_state(self, inputs):
        left, right, softdrop, harddrop, cwspin, ccwspin, hold = inputs
        self.reward = 0
        harddropbypass = 0
        #hold
        if hold and not self.holdstate:
            for position in self.currentpositions:
                self.board[position] = 0
            if not self.holdpiece:
                self.holdpiece = self.currentpiece
            else:
                self.holdpiece, self.currentpiece = self.currentpiece, self.holdpiece
                self.nextsequence.insert(0, self.currentpiece)
            self.advancepiece()
            self.holdstate = True
            return False

        #movement 
        if left:
            self.leftdas += 1
            if self.leftdas != 2:
                self.leftmove()
        else:
            self.leftdas = 0
        if right:
            self.rightdas += 1
            if self.rightdas != 2:
                self.rightmove()
        else:
            self.rightdas = 0

        #rotation
        if cwspin and not self.cwstate:
            self.rotatecw()
            self.cwstate = True
        elif not cwspin:
            self.cwstate = False
        if ccwspin and not self.ccwstate:
            self.rotateccw()
            self.ccwstate = True
        elif not ccwspin:
            self.ccwstate = False

        #soft drop
        if softdrop:
            self.softdrop()

        self.tickcount += 1
        if self.tickcount % 5 == 0:
            prevpositions = self.currentpositions
            self.softdrop()
            if prevpositions == self.currentpositions:
                harddropbypass = 1
        #if self.tickcount % 1000 == 0:
        #    print(self.tickcount)

        #hard drop
        if (harddrop and not self.harddropstate) or harddropbypass:
            self.harddrop()
            if not harddropbypass:
                self.reward += 1
            #death
            for y in range(1, 3):
                for x in range(6, 10):
                    if self.board[x, y] != 0:
                        #print("loss")
                        return True
            #line clear
            deletions = []
            for line in range(23):
                if self.board[:,line].all():
                    deletions.append(line)
            self.board = np.delete(self.board, deletions, axis=1)
            for _ in range(len(deletions)):
                self.board = np.insert(self.board, 0, [10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10], axis=1)
            if len(deletions) == 1:
                self.reward += 3
                self.lastclear = "single"
            if len(deletions) == 2:
                self.reward += 9
                self.lastclear = "double"
            if len(deletions) == 3:
                self.reward += 27
                self.lastclear = "triple"
            if len(deletions) == 4:
                self.reward += 81
                self.lastclear = "tetris"
            self.advancepiece()
            self.harddropstate = True
            self.holdstate = False
        elif not harddrop:
            self.lastclear = ""
            self.harddropstate = False
        #check for loss
        self.score += self.reward
        #print(self.score)
        return False

    def get_move(self):
        kbstate = pygame.key.get_pressed()
        left = kbstate[K_LEFT]
        right = kbstate[K_RIGHT]
        softdrop = kbstate[K_DOWN]
        harddrop = kbstate[K_SPACE]
        cwspin = kbstate[K_x]
        ccwspin = kbstate[K_z]
        hold = kbstate[K_c]
        return left, right, softdrop, harddrop, cwspin, ccwspin, hold

    def get_score(self):
        return self.score

    def get_reward(self):
        return self.reward

    def play_game(self):
        loss = False
        self.advancepiece()
        while not loss:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
            #print(self.get_move())
            loss = self.update_state(self.get_move())
            self.draw_board()
            #print(self.get_score())
            self.clock.tick(20)
        print("loss")

    def get_state(self):
        state = self.board[3:13, 0:23].flatten().tolist() + self.nextsequence[0:6] + [self.holdpiece, int(self.holdstate), int(self.harddropstate), self.leftdas, self.rightdas, int(self.cwstate), int(self.ccwstate)]
        return np.array(state)

    def reset(self):
        self.init()
        self.advancepiece()
        return self.get_state()

    def step(self, action):
        inputs = [0] * 7
        inputs[action] = 1
        done = self.update_state(tuple(inputs))
        info = {}
        if self.lastclear:
            info[0] = self.lastclear
        return self.get_state(), self.get_reward(), done, info

    def play_bot(self):
        model = keras.models.load_model("my_model.h5")
        loss = False
        self.advancepiece()
        while not loss:
            state = self.get_state()
            inputs = model.predict(np.expand_dims(np.array(state), axis=0))
            inputs = np.around(inputs, decimals=0)[0]
            loss = self.update_state(tuple(inputs))
            self.draw_board()
            #print(self.get_score())
            self.clock.tick(20)
        print("loss")

    def render(self, mode="human", close=False):
        pass

def main():
    t = Tetris(True)
    t.play_game()

if __name__ == "__main__":
    main()
    
