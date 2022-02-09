
from gym_combat.gym_combat.envs.Arena.AbsDecisionMaker import AbsDecisionMaker
from gym_combat.gym_combat.envs.Common.constants import np, SIZE_H, SIZE_W, DSM, AgentAction, NUMBER_OF_ACTIONS, FIRE_RANGE, BB_MARGIN



class CPoint:

    def __init__(self, h, w):

        self.h = h
        self.w = w

class Entity:
    def __init__(self, decision_maker: AbsDecisionMaker = None):

        self._decision_maker = decision_maker #not in use due to training with stable_baselines3
        self.h = -1
        self.w = -1

    def _choose_random_position(self):
        is_obs = True
        while is_obs:
            self.h = np.random.randint(0, SIZE_H)
            self.w = np.random.randint(0, SIZE_W)
            is_obs = self.is_obs(self.h, self.w)


    def __str__(self):
        # for debugging purposes
        return f"{self.h}, {self.w}"

    def __sub__(self, other):
        return (self.h - other.h, self.w - other.w)

    def __add__(self, other):
        return (self.h + other.h, self.w + other.w)

    def get_coordinates(self):
        return self.h, self.w

    def set_coordinatess(self, h, w):
        self.h = h
        self.w = w

    def move(self, h=None, w=None):
        # if no value for h- move randomly
        if h==None:
            new_h = self.h + np.random.randint(-1, 2)
        else:
            new_h = self.h + h

        # if no value for w- move randomly
        if w==None:
            new_w = self.w + np.random.randint(-1, 2)
        else:
            new_w = self.w + w

        is_legal, new_h, new_w = self.check_if_move_is_legal(new_h, new_w)
        if is_legal:
            self.h = new_h
            self.w = new_w
        # else: stay at last spot

    def check_if_move_is_legal(self, h, w):
        # if we are out of bounds- fix!
        if h < 0:
            h = 0
        elif h > SIZE_H - 1:
            h = SIZE_H - 1
        if w < 0:
            w = 0
        elif w > SIZE_W - 1:
            w = SIZE_W - 1

        is_obs = self.is_obs(h, w)

        return not(is_obs), h, w

    def is_obs(self, h=-1, w=-1):
        # if first draw of start point
        if h == -1 and w == -1:
            return True

        if DSM[h,w] == 1:
            return True
        return False

    def action(self, a: AgentAction):
        if NUMBER_OF_ACTIONS==9:
            """9 possible moves!"""
            # BUG: there is a coordinate switch. Here we changed xs and ys places to fix it.
            if a == AgentAction.TopRight: #1
                self.move(h=-1, w=1)
            elif a == AgentAction.Right: #2
                self.move(h=0, w=1)
            elif a == AgentAction.BottomRight: #3
                self.move(h=1, w=1)
            elif a == AgentAction.Top: # 4
                self.move(h=-1, w=0)
            elif a == AgentAction.Stay:  # 5 - stay in place!
                self.move(h=0, w=0)
            elif a == AgentAction.Bottom: # 6
                self.move(h=1, w=0)
            elif a == AgentAction.TopLeft : # 7
                self.move(h=-1, w=-1)
            elif a==AgentAction.Left: #8
                self.move(h=0, w=-1)
            elif a == AgentAction.BottomLeft: #0
                self.move(h=1, w=-1)

        else:
            """4 possible moves!"""
            if a == AgentAction.Right: #0
                self.move(h=0, w=1)
            elif a == AgentAction.Top: #1
                self.move(h=-1, w=0)
            elif a == AgentAction.Bottom: #2
                self.move(h=1, w=0)
            elif a == AgentAction.Left: #3
                self.move(h=0, w=-1)

