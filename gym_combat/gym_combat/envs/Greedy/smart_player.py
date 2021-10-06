import networkx as nx
from gym_combat.gym_combat.envs.Arena.CState import State
from gym_combat.gym_combat.envs.Arena.AbsDecisionMaker import AbsDecisionMaker
from gym_combat.gym_combat.envs.Common.constants import *
import numpy as np
import os
import gym_combat.gym_combat.envs.Greedy.smart_player_map as spm

PRINT_FLAG = False


class SmartPlayer(AbsDecisionMaker):
    def __init__(self, UPDATE_CONTEXT=True , path_model_to_load=None):

        self._action = -1
        self._type = AgentType.Smart
        self.episode_number = 0
        self._epsilon = 0
        self.path_model_to_load = None
        self.path_planner = spm.PathPlanner()

        self.G = self.create_graph()

        self.add_to_all_pairs_distances = False
        self.add_to_all_pairs_shortest_path = False
        self.add_to_closest_target_dict = False
        self.all_pairs_distances = {}
        self.all_pairs_shortest_path = {}
        self.closest_target_dict = {}
        self.load_data()

    def load_data(self):
        pass

    def create_graph(self):
        pass

    def set_initial_state(self, state: State, episode_number, input_epsilon=None):
        pass

    def update_context(self, state: State, action : AgentAction, new_state: State, reward, is_terminal, EVALUATE=True):
        pass

    def get_action(self, state: State, evaluate=False)-> AgentAction:
        action = self.path_planner.plan_next_action(state)
        self._action = action
        return self._action

    def find_closest_point_in_enemy_LOS(self, my_pos, enemy_pos):
        pass


    def find_path_to_closest_target(self, my_pos, closest_target):
        pass

    def get_action_9_actions(self, delta_h, delta_w):
        """9 possible moves!"""
        if delta_w == 1 and delta_h == -1:
            a = AgentAction.TopRight
        elif delta_w == 1 and delta_h == 0:
            a = AgentAction.Right
        elif delta_w == 1 and delta_h == 1:
            a = AgentAction.BottomRight
        elif delta_w == 0 and delta_h == -1:
            a = AgentAction.Top
        elif delta_w == 0 and delta_h == 0:
            a = AgentAction.Stay
        elif delta_w == 0 and delta_h == 1:
            a = AgentAction.Bottom
        elif delta_w == -1 and delta_h == -1:
            a = AgentAction.TopLeft
        elif delta_w == -1 and delta_h == 0:
            a = AgentAction.Left
        elif delta_w == -1 and delta_h == 1:
            a = AgentAction.BottomLeft

        return a

    def get_action_4_actions(self, delta_h, delta_w):
        """4 possible moves!"""
        if delta_w == 1 and delta_h == 0:
            a = AgentAction.Right
        elif delta_w == 0 and delta_h == 1:
            a = AgentAction.Bottom
        elif delta_w == 0 and delta_h == -1:
            a = AgentAction.Top
        elif delta_w == -1 and delta_h == 0:
            a = AgentAction.Left
        return a

    def type(self) -> AgentType:
        return self._type

    def get_epsolon(self):
        return self._epsilon

    def save_model(self, episodes_rewards, save_folder_path, color):
        pass

    def calc_all_pairs_data(self, DSM):
        pass


    def remove_data_obs(self, DSM):
        pass


if __name__ == '__main__':
    #PRINT_FLAG = True
    from PIL import Image
    import cv2
    srcImage = Image.open("../Common/maps/Berlin_1_256.png")

    img1 = np.array(srcImage.convert('L').resize((100, 100)))
    img2 = cv2.bitwise_not(img1)
    obsticals = cv2.inRange(img2, 250, 255)
    c, _ = cv2.findContours(obsticals, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thicken_obs_and_edges = cv2.drawContours(obsticals, c, -1, (255, 255, 255), 2)
    thicken_obs_and_edges[thicken_obs_and_edges > 0] = 1
    DSM = thicken_obs_and_edges

    if False:
        plt.matshow(thicken_obs_and_edges)
        plt.show(DSM)

    GP = smart_player()
    #GP.remove_data_obs(DSM)
    #GP.calc_all_pairs_data(DSM)

    blue_pos = Position(3, 10)
    red_pos = Position(10, 3)
    ret_val = State(my_pos=blue_pos, enemy_pos=red_pos)
    #
    a = GP.get_action(ret_val)
    print("The action to take is: ", a)
