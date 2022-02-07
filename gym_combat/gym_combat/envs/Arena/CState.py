from gym_combat.gym_combat.envs.Arena.Position import Position
from gym_combat.gym_combat.envs.Common.constants import *
from gym_combat.gym_combat.envs.Arena.helper_funcs import check_if_LOS
import matplotlib.pyplot as plt


class State(object):

    def __init__(self, my_pos: Position, enemy_pos, Red_won=False, number_of_steps_blue_stay=0,
                 number_of_steps_red_stay=0, whos_turn=None):
        self.Red_won = Red_won
        self.my_pos = my_pos
        self.enemy_pos = enemy_pos
        self.number_of_steps_blue_stay = number_of_steps_blue_stay
        self.number_of_steps_red_stay = number_of_steps_red_stay
        self.whos_turn = whos_turn

        self.env = np.zeros((SIZE_H, SIZE_W, 3), dtype=np.uint8)  # starts an rbg of small world
        self.img = self.get_image()

    def get_image(self):
        self.env = np.zeros((SIZE_H, SIZE_W, 3), dtype=np.uint8)  # starts an rbg of small world
        if self.enemy_pos is not None:
            points_in_enemy_los_as_tuple = DICT_POS_LOS_TUPLE[(self.enemy_pos._x, self.enemy_pos._y)]
            self.env[points_in_enemy_los_as_tuple] = dict_of_colors_for_state[DARK_DARK_RED_N]

            # set danger zone in state
            if NONEDETERMINISTIC_TERMINAL_STATE:
                points_in_enemy_fire_range = DICT_POS_FIRE_RANGE[(self.enemy_pos._x, self.enemy_pos._y)]
                enemy_color = dict_of_colors_for_state[RED_N]
                for point in points_in_enemy_fire_range:
                    dist = np.linalg.norm(np.array(point) - np.array([self.enemy_pos._x, self.enemy_pos._y]))
                    if self.whos_turn is Color.Blue:
                        dist = np.max([0, dist - self.number_of_steps_red_stay])
                    elif self.whos_turn is Color.Red:
                        dist = np.max([0, dist - self.number_of_steps_blue_stay])
                    dist_floor = np.floor(dist)
                    color = tuple(map(lambda i, j: int(i - j), enemy_color, (15 * dist_floor, 0, 0)))
                    self.env[point[0]][point[1]] = color
            else:
                points_in_enemy_fire_range_tuple = DICT_POS_FIRE_RANGE_TUPLE[(self.enemy_pos._x, self.enemy_pos._y)]
                self.env[points_in_enemy_fire_range_tuple] = dict_of_colors_for_state[DARK_RED_N]
            # set enemy (red player)
            self.env[self.enemy_pos._x, self.enemy_pos._y] = dict_of_colors_for_state[RED_N]

        if not self.Red_won:
            # set blue player
            self.env[self.my_pos._x, self.my_pos._y] = dict_of_colors_for_state[BLUE_N]

        # set obstacles
        self.env[np.where(DSM == 1)] = dict_of_colors_for_graphics[GREY_N]

        if (BB_STATE):
            extended_env = np.zeros((SIZE_H + 2 * BB_EXTENSION, SIZE_W + 2 * BB_EXTENSION, 3), dtype=np.uint8)
            extended_env[BB_EXTENSION:-BB_EXTENSION, BB_EXTENSION: - BB_EXTENSION] = self.env
            BB_env = extended_env[self.my_pos._x: self.my_pos._x + 2 * BB_EXTENSION + 1,
                                  self.my_pos._y: self.my_pos._y + 2 * BB_EXTENSION + 1]

            if False:
                plt.matshow(self.env)
                plt.show()
                plt.matshow(BB_env)
                plt.show()

            return BB_env
        return self.env


def print_env(env):
    # print state for debug
    import matplotlib.pyplot as plt
    plt.matshow(env)
    plt.show()