import matplotlib.pyplot as plt
import networkx as nx
from gym_combat.gym_combat.envs.Arena.Position import Position
from gym_combat.gym_combat.envs.Arena.CState import State
from gym_combat.gym_combat.envs.Arena.AbsDecisionMaker import AbsDecisionMaker
from gym_combat.gym_combat.envs.Common.constants import *
import numpy as np
import os
PRINT_FLAG = False


class Greedy_player(AbsDecisionMaker):

    def __init__(self, UPDATE_CONTEXT=True , path_model_to_load=None):

        self._action = -1
        self._type = AgentType.Greedy
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

        all_pairs_shortest_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_shortest_path_' + DSM_name + '_' + str(FIRE_RANGE) + '.pkl'
        if os.path.exists(all_pairs_shortest_path):
            with open(all_pairs_shortest_path, 'rb') as f:
                self.all_pairs_shortest_path = pickle.load(f)
                print("Greedy: all_pairs_shortest_path loaded")

        closest_target_dict_path = 'gym_combat/gym_combat/envs/Greedy/closest_target_dict_' + DSM_name + '_' + str(FIRE_RANGE) + '.pkl'
        if os.path.exists(closest_target_dict_path):
            with open(closest_target_dict_path, 'rb') as f:
                self.closest_target_dict = pickle.load(f)
                print("Greedy: closest_target_dict loaded")

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

        # self.all_pairs_distances = dict(nx.floyd_warshall(G))
        # self.all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(G))

        # nx.write_gpickle(G, 'Greedy_'+DSM_name+'.pkl')
        # loaded_G = nx.read_gpickle('Greedy_'+DSM_name+'.pkl')


        if PRINT_FLAG:
            path = self.all_pairs_shortest_path[(3,10)][(10,3)]
            path_edges = set(zip(path, path[1:]))
            nx.draw_networkx(G, pos=pos, labels=labels, font_size=5, with_labels=True, node_size=50)

            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='g')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color='b')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[1]], node_color='black')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color='r')
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='black', width=3)
            plt.axis('equal')
            plt.show()

        return G

    def set_initial_state(self, state: State, episode_number, input_epsilon=None):
        pass


    def update_context(self, state: State, action : AgentAction, new_state: State, reward, is_terminal, EVALUATE=True):

        pass


    def get_action(self, state: State, evaluate=False)-> AgentAction:
        my_pos = state.my_pos.get_tuple()
        enemy_pos = state.enemy_pos.get_tuple()


        closest_target = self.find_closest_point_in_enemy_LOS(my_pos, enemy_pos)


        if closest_target is None or closest_target == np.inf:
            # return random move
            a = AgentAction(np.random.randint(0, NUMBER_OF_ACTIONS))
            return a

        path_to_closest_target = self.find_path_to_closest_target(my_pos, closest_target)



        #path_to_closest_target = nx.shortest_path(self.G, my_pos, closest_target)
        #path_to_closest_target = self.all_pairs_shortest_path[my_pos][closest_target]
        if len(path_to_closest_target)<=1:
            # return random move
            a = AgentAction(np.random.randint(0, NUMBER_OF_ACTIONS))
            return a

        first_step = path_to_closest_target[1]
        delta_h =  first_step[0]-my_pos[0]
        delta_w = first_step[1] - my_pos[1]

        a=-1
        if NUMBER_OF_ACTIONS==9:
            a = self.get_action_9_actions(delta_h, delta_w)
        else:
            a = self.get_action_4_actions(delta_h, delta_w)

        if PRINT_FLAG:
            # print graph for debug
            pos = dict((n, n) for n in self.G.nodes())  # Dictionary of all positions
            labels = dict(((i, j), (i, j)) for i, j in self.G.nodes())
            path_to_closest_enemy_los = path_to_closest_target
            points_in_enemy_los = DICT_POS_FIRE_RANGE[enemy_pos]
            points_in_enemy_los_edges = set(zip(path_to_closest_target, path_to_closest_target[1:]))
            nx.draw_networkx(self.G, pos=pos, labels=labels, font_size=5, with_labels=True, node_size=50)
            nx.draw_networkx_nodes(self.G, pos, nodelist=points_in_enemy_los, node_color='r')
            nx.draw_networkx_nodes(self.G, pos, nodelist=path_to_closest_enemy_los, node_color='g')
            nx.draw_networkx_nodes(self.G, pos, nodelist=[my_pos], node_color='b')
            nx.draw_networkx_nodes(self.G, pos, nodelist=[first_step], node_color='black')
            nx.draw_networkx_nodes(self.G, pos, nodelist=[closest_target], node_color='y')
            nx.draw_networkx_edges(self.G, pos, edgelist=points_in_enemy_los_edges, edge_color='black', width=3)
            plt.axis('equal')
            plt.show()

        self._action = a

        return self._action

    def find_closest_point_in_enemy_LOS(self, my_pos, enemy_pos):
        # all potential targets
        dist_pos_enemy = np.linalg.norm(np.array(my_pos) - np.array(enemy_pos))
        if dist_pos_enemy<=FIRE_RANGE:
            return enemy_pos

        points_in_enemy_los = DICT_POS_FIRE_RANGE[enemy_pos]

        # find closest point in enemy line of sight
        if my_pos in self.closest_target_dict.keys():
            if enemy_pos in self.closest_target_dict[my_pos].keys():
                closest_target = self.closest_target_dict[my_pos][enemy_pos]
                return closest_target
        else:
            self.closest_target_dict[my_pos] = {}


        best_distance = np.inf
        closest_target = None


        for point in points_in_enemy_los:
            if not (my_pos in self.all_pairs_distances.keys()):
                self.all_pairs_distances[my_pos] = {}
            if not (point in self.all_pairs_distances.keys()):
                self.all_pairs_distances[point] = {}

            if not (point in self.all_pairs_distances[my_pos].keys()):
                if DSM[my_pos] == 1 or DSM[point] == 1 or (not nx.has_path(self.G, my_pos, point)):
                    dist_me_point = np.Inf
                else:
                    dist_me_point = nx.shortest_path_length(self.G, my_pos, point)
                self.all_pairs_distances[my_pos][point] = dist_me_point
                self.all_pairs_distances[point][my_pos] = dist_me_point
                self.add_to_all_pairs_distances = True
            else:
                dist_me_point = self.all_pairs_distances[my_pos][point]


            if dist_me_point < best_distance:
                best_distance = dist_me_point
                closest_target = point

        self.closest_target_dict[my_pos][enemy_pos] = closest_target
        self.add_to_closest_target_dict = True
        return closest_target


    def find_path_to_closest_target(self, my_pos, closest_target):

        if my_pos in self.all_pairs_shortest_path.keys():
            if closest_target in self.all_pairs_shortest_path[my_pos].keys():
                path_to_closest_target = self.all_pairs_shortest_path[my_pos][closest_target]
                return path_to_closest_target

        if closest_target in self.all_pairs_shortest_path.keys():
            if my_pos in self.all_pairs_shortest_path[closest_target].keys():
                path_to_closest_target_reversed = self.all_pairs_shortest_path[closest_target][my_pos]
                return list(path_to_closest_target_reversed.__reversed__())

        #print("first time calc:    my_pos: ", my_pos, ", closest_target: ", closest_target)
        if not (my_pos in self.all_pairs_shortest_path.keys()):
            self.all_pairs_shortest_path[my_pos] = {}

        if DSM[my_pos] == 1 or DSM[closest_target] == 1 or (not nx.has_path(self.G, my_pos, closest_target)):
            path_to_closest_target = []
            self.all_pairs_shortest_path[my_pos][closest_target] = path_to_closest_target
            self.add_to_all_pairs_shortest_path = True
        else:
            path_to_closest_target = nx.shortest_path(self.G, my_pos, closest_target)
            self.all_pairs_shortest_path[my_pos][closest_target] = path_to_closest_target
            self.add_to_all_pairs_shortest_path = True

        #print("dist is: ", len(path_to_closest_target))
        return path_to_closest_target

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
        if self.add_to_all_pairs_distances:
            with open('gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '_' + str(FIRE_RANGE) + '.pkl', 'wb') as f:
                pickle.dump(self.all_pairs_distances, f,  protocol=2)
                self.add_to_all_pairs_distances = False
        if self.add_to_all_pairs_shortest_path:
            with open('gym_combat/gym_combat/envs/Greedy/all_pairs_shortest_path_' + DSM_name + '_' + str(FIRE_RANGE) + '.pkl', 'wb') as f:
                pickle.dump(self.all_pairs_shortest_path, f,  protocol=2)
                self.add_to_all_pairs_shortest_path = False
        if self.add_to_closest_target_dict:
            with open('gym_combat/gym_combat/envs/Greedy/closest_target_dict_' + DSM_name + '_' + str(FIRE_RANGE) + '.pkl', 'wb') as f:
                pickle.dump(self.closest_target_dict, f,  protocol=2)
                self.add_to_closest_target_dict = False

    def calc_all_pairs_data(self, DSM):
        SIZE_W = 100
        SIZE_H = 100
        G = nx.grid_2d_graph(SIZE_W, SIZE_H)

        if NUMBER_OF_ACTIONS >= 8:
            Diagonals_Weight = 1
            # add diagonals edges
            G.add_edges_from([
                                 ((x, y), (x + 1, y + 1))
                                 for x in range(SIZE_H - 1)
                                 for y in range(SIZE_H - 1)
                             ] + [
                                 ((x + 1, y), (x, y + 1))
                                 for x in range(SIZE_H - 1)
                                 for y in range(SIZE_H - 1)
                             ], weight=Diagonals_Weight)

        # remove obstacle nodes and edges
        for x in range(SIZE_W):
            for y in range(SIZE_H):
                if DSM[x][y] == 1.:
                    G.remove_node((x, y))

        # nx.write_gpickle(G, 'G_' + DSM_name + '.pkl')


        print("starting all_pairs_distances")
        all_pairs_distances = dict(nx.all_pairs_shortest_path_length(G))

        with open('gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '_' + 'regular_pickle' + '.pkl',
                  'wb') as f:
            pickle.dump(all_pairs_distances, f, protocol=2)
            print("finished all_pairs_distances: pickle.dump(all_pairs_distances, f, protocol=2)")


        import bz2
        with bz2.open('all_pairs_distances_' + DSM_name + '___' + '.pkl', "wb") as f:
            f.write(all_pairs_distances)
        # sfile = bz2.BZ2File('all_pairs_distances_' + DSM_name + '___' + '.pkl', 'wb', 'w')
        # pickle.dump(all_pairs_distances, sfile)
        print("finished all_pairs_distances")

        with bz2.open('all_pairs_distances_' + DSM_name + '___' + '.pkl', "rb") as f:
            # Decompress data from file
            content = f.read()
        print("dist from (5,5) to (5,6) is: ", content[(5,5)][(5,5)])

        # hugeData = {'key': {'x': 1, 'y': 2}}
        # with contextlib.closing(bz2.BZ2File('data.json.bz2', 'wb')) as f:
        #     json.dump(hugeData, f)
        #
        # with open('all_pairs_shortest_path_' + DSM_name + '___' + '.pkl', 'wb') as f:
        #     pickle.dump(all_pairs_shortest_path, f, protocol=2)
        # print("finished all_pairs_shortest_path")
        #
        #
        # #loaded_G = nx.read_gpickle('Greedy_'+DSM_name+'.pkl')


    def remove_data_obs(self, DSM):
        all_pairs_distances_path = 'all_pairs_distances_100X100_Berlin___.pkl'
        if os.path.exists(all_pairs_distances_path):
            with open(all_pairs_distances_path, 'rb') as f:
                all_pairs_distances = pickle.load(f)
                print("Greedy: all_pairs_distances_100X100_Berlin___ loaded")

        filtered_data = {}
        for p1 in all_pairs_distances.keys():
            for p2 in all_pairs_distances[p1].keys():
                if DSM[p1]==0 and DSM[p2]==0:
                    if not p1 in filtered_data.keys():
                        filtered_data[p1]={}
                    filtered_data[p1][p2] = all_pairs_distances[p1][p2]

        with open('all_pairs_distances_' + DSM_name + '___filtered' + '.pkl', 'wb') as f:
            pickle.dump(filtered_data, f,  protocol=2)


        short_pathes = {}
        counter=0
        for (x1,y1) in all_pairs_distances.keys():
            for (x2, y2) in all_pairs_distances[(x1,y1)].keys():
                dist = all_pairs_distances[(x1,y1)][(x2, y2)]
                if dist<FIRE_RANGE*4:
                    if not (x1,y1) in short_pathes.keys():
                        short_pathes[(x1,y1)]={}
                    short_pathes[(x1, y1)][(x2, y2)]=self.all_pairs_shortest_path.keys()
                else:
                    counter+=1
            print("done with (", x1, ", ", y1, ")")

        with open('./Greedy/all_pairs_shortest_path_' + DSM_name + '___filtered_long_paths' + '.pkl', 'wb') as f:
            pickle.dump(short_pathes, f,  protocol=2)


def calc_all_pairs_data():
    SIZE_W = 100
    SIZE_H = 100
    G = nx.grid_2d_graph(SIZE_W, SIZE_H)

    if NUMBER_OF_ACTIONS >= 8:
        Diagonals_Weight = 1
        # add diagonals edges
        G.add_edges_from([
                             ((x, y), (x + 1, y + 1))
                             for x in range(SIZE_H - 1)
                             for y in range(SIZE_H - 1)
                         ] + [
                             ((x + 1, y), (x, y + 1))
                             for x in range(SIZE_H - 1)
                             for y in range(SIZE_H - 1)
                         ], weight=Diagonals_Weight)

    # remove obstacle nodes and edges
    for x in range(SIZE_W):
        for y in range(SIZE_H):
            if DSM[x][y] == 1.:
                G.remove_node((x, y))

    # nx.write_gpickle(G, 'G_' + DSM_name + '.pkl')

    all_pairs_distances_path = 'all_pairs_distances_' + DSM_name + '___' + '.pkl'
    if os.path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")
    else:

        print("starting all_pairs_distances")
        all_pairs_distances = dict(nx.all_pairs_shortest_path_length(G))

        with open('gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '_' + 'pickle'+ '.pkl',
                  'wb') as f:
            pickle.dump(all_pairs_distances, f, protocol=2)
            print("finished all_pairs_distances: pickle.dump(all_pairs_distances, f, protocol=2)")

    # print("starting all_pairs_shortest_path")
    # all_pairs_shortest_path = dict(nx.all_pairs_shortest_path(G, cutoff=75))
    #
    #
    # with open('gym_combat/gym_combat/envs/Greedy/all_pairs_shortest_path' + DSM_name + '_' + '>75'+ '.pkl',
    #           'wb') as f:
    #     pickle.dump(all_pairs_shortest_path, f, protocol=2)
    #     print("finished all_pairs_shortest_path: pickle.dump(all_pairs_shortest_path, f, protocol=2)")

    SIZE_W = 10
    SIZE_H = 10
    all_pairs_shortest_path_less_than_5_no_double = {}
    for x1 in range(0, SIZE_W):
        for y1 in range(0, SIZE_H):
            print("starting ", str(x1), " ", str(y1))
            all_pairs_shortest_path_less_than_5_no_double[(x1, y1)] = {}
            if DSM[x1][y1]==1:
                continue
            for x2 in range(0, SIZE_W):
                for y2 in range(0, SIZE_H):
                    if not DSM[x2][y2]==1:
                        if (x1, y1) in all_pairs_distances.keys() and (x2, y2) in all_pairs_distances[(x1, y1)].keys():
                            print(x1, y1, x2, y2)
                            if all_pairs_distances[(x1, y1)][(x2, y2)]<5:
                                if (x2, y2) in all_pairs_shortest_path_less_than_5_no_double.keys() and (x1, y1) not in all_pairs_shortest_path_less_than_5_no_double[(x2, y2)]:
                                    path = nx.astar_path(G, (x1, y1), (x2, y2))
                                    all_pairs_shortest_path_less_than_5_no_double[(x1, y1)][(x2, y2)] = path

    with open('all_pairs_shortest_path' + DSM_name + '_' + '5'+ '.pkl',
              'wb') as f:
        pickle.dump(all_pairs_shortest_path_less_than_5_no_double, f, protocol=2)
        print("finished all_pairs_shortest_path_less_than_75_no_double: pickle.dump(all_pairs_shortest_path_less_than_75_no_double, f, protocol=2)")



    # import bz2
    # with bz2.open('all_pairs_distances_' + DSM_name + '___' + '.pkl', "wb") as f:
    #     f.write(all_pairs_distances)
    # # sfile = bz2.BZ2File('all_pairs_distances_' + DSM_name + '___' + '.pkl', 'wb', 'w')
    # # pickle.dump(all_pairs_distances, sfile)
    # print("finished all_pairs_distances")
    #
    # with bz2.open('all_pairs_distances_' + DSM_name + '___' + '.pkl', "rb") as f:
    #     # Decompress data from file
    #     content = f.read()
    # print("dist from (5,5) to (5,6) is: ", content[(5,5)][(5,5)])


if __name__ == '__main__':
    # #PRINT_FLAG = True
    # from PIL import Image
    # import cv2
    # srcImage = Image.open("../Common/maps/Berlin_1_256.png")
    #
    # img1 = np.array(srcImage.convert('L').resize((100, 100)))
    # img2 = cv2.bitwise_not(img1)
    # obsticals = cv2.inRange(img2, 250, 255)
    # c, _ = cv2.findContours(obsticals, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # thicken_obs_and_edges = cv2.drawContours(obsticals, c, -1, (255, 255, 255), 2)
    # thicken_obs_and_edges[thicken_obs_and_edges > 0] = 1
    # DSM = thicken_obs_and_edges
    #
    # if False:
    #     plt.matshow(thicken_obs_and_edges)
    #     plt.show(DSM)
    #
    # GP = Greedy_player()
    # #GP.remove_data_obs(DSM)
    # GP.calc_all_pairs_data(DSM)
    #
    # # blue_pos = Position(3, 10)
    # # red_pos = Position(10, 3)
    # # ret_val = State(my_pos=blue_pos, enemy_pos=red_pos)
    # #    # a = GP.get_action(ret_val)
    # # print("The action to take is: ", a)


    calc_all_pairs_data()