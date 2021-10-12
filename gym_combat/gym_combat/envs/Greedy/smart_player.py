import networkx as nx
from gym_combat.gym_combat.envs.Arena.CState import State
from gym_combat.gym_combat.envs.Arena.AbsDecisionMaker import AbsDecisionMaker
from gym_combat.gym_combat.envs.Common.constants import *
import numpy as np
import os

PRINT_FLAG = False


class SmartPlayer(AbsDecisionMaker):
    def __init__(self, UPDATE_CONTEXT=True , path_model_to_load=None):

        self._action = -1
        self._type = AgentType.Smart
        self.episode_number = 0
        self._epsilon = 0
        self.path_model_to_load = None

        self.G = self.create_graph()

        self.add_to_all_pairs_distances = False
        self.add_to_all_pairs_shortest_path = False
        self.add_to_closest_target_dict = False
        self.all_pairs_distances = {}
        self.all_pairs_shortest_path = {}
        self.closest_target_dict = {}
        self.load_data()

    def load_data(self):
        self.all_pairs_distances = all_pairs_distances

        all_pairs_shortest_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_shortest_path_' + DSM_name + '_' + str(
            FIRE_RANGE) + '.pkl'
        if os.path.exists(all_pairs_shortest_path):
            with open(all_pairs_shortest_path, 'rb') as f:
                self.all_pairs_shortest_path = pickle.load(f)
                print("Smart: all_pairs_shortest_path loaded")
        covers_map = 'gym_combat/gym_combat/envs/Common/Preprocessing/covers_map_100x100_Berlin.pkl'
        if os.path.exists(covers_map):
            with open(covers_map, 'rb') as f:
                self.maps_map = pickle.load(f)
                print("Smart: covers map loaded")


    def create_graph(self):
        G = nx.grid_2d_graph(SIZE_H, SIZE_W)
        pos = dict((n, n) for n in G.nodes())  # Dictionary of all positions
        labels = dict(((i, j), (i, j)) for i, j in G.nodes())

        if NUMBER_OF_ACTIONS >= 8:
            Diagonals_Weight = 1
            # add diagonals edges
            G.add_edges_from([
                                 ((h, w), (h + 1, w + 1))
                                 for h in range(SIZE_H - 1)
                                 for w in range(SIZE_W - 1)
                             ] + [
                                 ((h + 1, w), (h, w + 1))
                                 for h in range(SIZE_H - 1)
                                 for w in range(SIZE_W - 1)
                             ], weight=Diagonals_Weight)

        # remove obstacle nodes and edges
        for h in range(SIZE_H):
            for w in range(SIZE_W):
                if DSM[h,w] == 1.:
                    G.remove_node((h, w))
        return G

    def set_initial_state(self, state: State, episode_number, input_epsilon=None):
        pass

    def update_context(self, state: State, action : AgentAction, new_state: State, reward, is_terminal, EVALUATE=True):
        pass

    def get_action(self, state: State, evaluate=False)-> AgentAction:
        action = self.plan_next_action(state)
        self._action = action
        return self._action

    def find_closest_point_in_enemy_LOS(self, my_pos, enemy_pos):
        pass


    def find_distance_to_target(self, my_pos, target):
        if not (my_pos in self.all_pairs_distances.keys()):
            self.all_pairs_distances[my_pos] = {}
        if not (target in self.all_pairs_distances.keys()):
            self.all_pairs_distances[target] = {}

        if not (target in self.all_pairs_distances[my_pos].keys()):
            if DSM[my_pos] == 1 or DSM[target] == 1 or (not nx.has_path(self.G, my_pos, target)):
                dist = np.Inf
            else:
                dist = nx.shortest_path_length(self.G, my_pos, target)
            self.all_pairs_distances[my_pos][target] = dist
            self.all_pairs_distances[target][my_pos] = dist
            self.add_to_all_pairs_distances = True
        else:
            dist = self.all_pairs_distances[my_pos][target]
        return dist


    def find_path_to_target(self, my_pos, target):
        if my_pos in self.all_pairs_shortest_path.keys():
            if target in self.all_pairs_shortest_path[my_pos].keys():
                path_to_target = self.all_pairs_shortest_path[my_pos][target]
                return path_to_target

        if target in self.all_pairs_shortest_path.keys():
            if my_pos in self.all_pairs_shortest_path[target].keys():
                path_to_target_reversed = self.all_pairs_shortest_path[target][my_pos]
                return list(path_to_target_reversed.__reversed__())

        #print("first time calc:    my_pos: ", my_pos, ", target: ", target)
        if not (my_pos in self.all_pairs_shortest_path.keys()):
            self.all_pairs_shortest_path[my_pos] = {}

        if DSM[my_pos] == 1 or DSM[target] == 1 or (not nx.has_path(self.G, my_pos, target)):
            path_to_target = []
            self.all_pairs_shortest_path[my_pos][target] = path_to_target
            self.add_to_all_pairs_shortest_path = True
        else:
            path_to_target = nx.shortest_path(self.G, my_pos, target)
            self.all_pairs_shortest_path[my_pos][target] = path_to_target
            self.add_to_all_pairs_shortest_path = True

        return path_to_target

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

    def plan_next_action(self, state):
        my_pos = state.my_pos.get_tuple()
        enemy_pos = state.enemy_pos.get_tuple()
        neighbors = np.asarray([[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1], [0,0]])

        # maybe fire is near and we can win:
        fire = DICT_POS_FIRE_RANGE[enemy_pos]
        for d in neighbors:
            if tuple(np.array(my_pos) + d) in fire:
                action = self.get_action_9_actions(d[0], d[1])
                return action
        # Else search for a cover:
        my_path = self.plan_path(my_pos, enemy_pos)
        if my_path:
            next_step = self.find_move_in_path(my_path)
            direc = np.asarray(next_step[:2]) - np.array(my_pos)
        else:
            # no cover, just run far from enemy:
            neighbors_dist = {}
            for d in neighbors:
                neighbors_dist[tuple(d)] = np.linalg.norm(np.array(my_pos) + d - np.array(enemy_pos), 1)
            for direc, dist in sorted(neighbors_dist.items(), key=lambda item: item[1], reverse=True):
                if DSM[tuple(np.array(my_pos) + direc)] == 0:
                    break

        action = self.get_action_9_actions(direc[0], direc[1])
        return action

    def plan_path(self, my_pos, enemy_pos):
        possible_locs = self.calc_possible_locs(enemy_pos, depth=FIRE_RANGE)
        possible_locs[possible_locs == 0] = FIRE_RANGE + 1

        covers_map = self.maps_map[enemy_pos[0], enemy_pos[1], :, :]

        closest_cover = self.select_cover(covers_map, my_pos, possible_locs, FIRE_RANGE)

        if len(closest_cover):
            if (np.asarray(my_pos) == closest_cover).all():
                return [(closest_cover[0], closest_cover[1])]
            return self.find_path_to_target(my_pos, closest_cover)
        else:
            return []

    def calc_possible_locs(self, enemy_loc, depth=10, neighborhood=8):
        queue = []
        queue.append(enemy_loc)
        res = (100 * DSM).astype(np.uint8)
        opponent_loc = np.asarray(enemy_loc)
        res[opponent_loc[0], opponent_loc[1]] = 2
        neighbors = np.asarray([[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]])
        if neighborhood == 4:
            neighbors = np.asarray([[0, -1], [0, 1], [-1, 0], [1, 0]])
        while queue:
            curr = queue.pop(0)
            for nei_dir in neighbors:
                nei = curr + nei_dir
                if nei[0] < 0 or nei[0] >= res.shape[0] or nei[1] < 0 or nei[1] >= res.shape[1]:
                    continue
                if res[nei[0], nei[1]]:
                    continue
                if res[curr[0], curr[1]] < depth:
                    res[nei[0], nei[1]] = res[curr[0], curr[1]] + 1
                    queue.append(nei)
                    # plt.imshow(res)

        return res

    def select_cover(self, covers_map, my_pos, possible_locs, depth=10):
        covers_dist_dict = {}
        for candidate_cover in zip(*np.where(covers_map > 0)):
            covers_dist_dict[candidate_cover] = self.find_distance_to_target(my_pos, candidate_cover)

        for cand, dist in sorted(covers_dist_dict.items(), key=lambda item: item[1]):
            if dist > depth:
                break
            reach_time = self.calc_reach_time(cand, possible_locs, depth)
            if dist < reach_time:
                return cand
        return []

    def calc_reach_time(self, cover, possible_locs, final_reach_time):
        for reach_time in range(2, final_reach_time):
            for possible_enemy_loc in zip(*np.where(possible_locs == reach_time)):
                if cover in DICT_POS_FIRE_RANGE[possible_enemy_loc]:
                    return reach_time
        return final_reach_time

    def find_move_in_path(self, player_path):
        if len(player_path) > 1:
            return player_path[1]
        else:
            return player_path[0]

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
