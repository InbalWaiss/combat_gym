
from gym_combat.gym_combat.envs.Common.constants import *
import pickle
import numpy as np
from gym_combat.gym_combat.envs.Arena.helper_funcs import check_if_LOS
import matplotlib.pyplot as plt
from gym_combat.gym_combat.envs.Arena.Environment import Environment
from gym_combat.gym_combat.envs.Arena.Entity import Entity

import os


def creat_and_save_dictionaries():
    los_from_pos_FIRE_RANGE = {}
    los_from_pos_LOS_RANGE = {}
    no_los_from_pos = {}
    for x1 in range(0, SIZE_W):
        for y1 in range(0, SIZE_H):
            los_from_pos_FIRE_RANGE[(x1, y1)] = []
            los_from_pos_LOS_RANGE[(x1, y1)] = []
            no_los_from_pos[(x1, y1)] = []
            if DSM[x1][y1]==1:
                continue
            for x2 in range(0, SIZE_W):
                for y2 in range(0, SIZE_H):
                    dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
                    if DSM[x2][y2] == 1 or dist>LOS_PENALTY_RANGE:
                        continue

                    is_los_LOS_PENALTY_RANGE, _ = check_if_LOS(x1, y1, x2, y2)
                    if is_los_LOS_PENALTY_RANGE and dist<=LOS_PENALTY_RANGE: #not obs and in LOS_PENALTY_RANGE
                        los_from_pos_LOS_RANGE[(x1, y1)].append((x2, y2))
                        if dist<=FIRE_RANGE:
                            los_from_pos_FIRE_RANGE[(x1, y1)].append((x2, y2))
                    else:
                        if DSM[x2, y2]!=1:
                            no_los_from_pos[(x1, y1)].append((x2, y2))

            print("finished ", x1, y1)

    save_obj(los_from_pos_FIRE_RANGE, "dictionary_position_los_"+DSM_name+'_'+str(FIRE_RANGE)+ '.pkl')
    save_obj(los_from_pos_LOS_RANGE, "dictionary_position_los_"+DSM_name+'_'+str(LOS_PENALTY_RANGE)+ '.pkl')
    save_obj(no_los_from_pos, "dictionary_position_no_los_"+DSM_name+'_'+str(LOS_PENALTY_RANGE)+ '.pkl')


def show_LOS_from_point(x1,y1):
    env = np.zeros((SIZE_W, SIZE_H, 3), dtype=np.uint8)  # starts an rbg of small world

    points_in_LOS = DICT_POS_LOS[(x1,y1)]
    for point in points_in_LOS:
        env[point[0]][point[1]] = dict_of_colors_for_graphics[BRIGHT_RED]

    env[x1][y1] = dict_of_colors_for_graphics[RED_N]

    for x in range(SIZE_W):
        for y in range(SIZE_H):
            if DSM[x][y] == 1.:
                env[x][y] = dict_of_colors_for_graphics[GREY_N]

    plt.matshow(env)
    plt.show()

def show_no_LOS_from_point(x1,y1):
    env = np.zeros((SIZE_W, SIZE_H, 3), dtype=np.uint8)  # starts an rbg of small world

    points_in_LOS = DICT_POS_NO_LOS[(x1,y1)]
    for point in points_in_LOS:
        env[point[0]][point[1]] = (100, 0, 0)

    env[x1][y1] = dict_of_colors_for_graphics[BLUE_N]

    for x in range(SIZE_W):
        for y in range(SIZE_H):
            if DSM[x][y] == 1.:
                env[x][y] = dict_of_colors_for_graphics[GREY_N]

    plt.matshow(env)
    plt.show()

def find_closest_point_not_in_los(x1,y1):
    arr = np.asarray(DICT_POS_NO_LOS[(x1, y1)])
    value = np.array([x1, y1])
    closest_point_no_loss = arr[np.linalg.norm(arr - value, axis=1).argmin()]
    env = np.zeros((SIZE_W, SIZE_H, 3), dtype=np.uint8)  # starts an rbg of small world
    points_in_LOS = DICT_POS_LOS[(x1, y1)]
    for point in points_in_LOS:
        env[point[0]][point[1]] = dict_of_colors_for_graphics[DARK_RED_N]
    env[x1][y1] = dict_of_colors_for_graphics[RED_N]
    env[closest_point_no_loss[0]][closest_point_no_loss[1]] = (200, 200, 200)
    for x in range(SIZE_W):
        for y in range(SIZE_H):
            if DSM[x][y] == 1.:
                env[x][y] = dict_of_colors_for_graphics[GREY_N]
    plt.matshow(env)
    plt.show()
    return closest_point_no_loss

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def calc_and_save_dominating_points():
    dominating_points_dict = {}
    for x1 in range(0, SIZE_W):
        for y1 in range(0, SIZE_H):
            goal_points = find_dominating_point(x1, y1)
            dominating_points_dict[(x1,y1)] = goal_points

    save_obj(dominating_points_dict, "dictionary_dominating_points")

def calc_and_save_lose_points():
    lose_points_dict = {}
    for x1 in range(0, SIZE_W):
        for y1 in range(0, SIZE_H):
            goal_points = find_lose_points(x1, y1)
            lose_points_dict[(x1,y1)] = goal_points

    save_obj(lose_points_dict, "dictionary_lose_points")


def find_dominating_point(x1, y1):
    DEBUG=False
    point1 = (x1, y1)
    arr = DICT_POS_LOS[(x1, y1)]
    goal_points = []
    for p in arr:
        if can_escape_by_one_step(point1, p):
            goal_points.append(p)
    if DEBUG:
        img_env = np.zeros((SIZE_W, SIZE_H, 3), dtype=np.uint8)  # starts an rbg of small world
        points_in_LOS = DICT_POS_LOS[(x1, y1)]
        for point in points_in_LOS:
            img_env[point[0]][point[1]] = dict_of_colors_for_graphics[DARK_RED_N]
        img_env[x1][y1] = dict_of_colors_for_graphics[RED_N]
        for x in range(SIZE_W):
            for y in range(SIZE_H):
                if DSM[x][y] == 1.:
                    img_env[x][y] = dict_of_colors_for_graphics[GREY_N]
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img_env)
        for point in goal_points:
            img_env[point[0]][point[1]] = dict_of_colors_for_graphics[GREEN_N]
        axs[1].imshow(img_env)
        plt.show()
    print("end ", x1, y1)
    return goal_points

def can_escape_by_one_step(point1, point2):
    env = Environment()

    env.red_player = Entity()
    env.red_player.x = point1[0]
    env.red_player.y = point1[1]

    env.blue_player = Entity()
    env.blue_player.x = point2[0]
    env.blue_player.y = point2[1]

    win_stat = env.compute_terminal()

    if win_stat==WinEnum.Blue:
        return True

    return False


def find_lose_points(x1, y1):
    DEBUG=True
    point1 = (x1, y1)
    arr = DICT_POS_LOS[(x1, y1)]
    goal_points = []
    for p in arr:
        if can_escape_by_one_step(p, point1):
            goal_points.append(p)
    if DEBUG:
        img_env = np.zeros((SIZE_W, SIZE_H, 3), dtype=np.uint8)  # starts an rbg of small world
        points_in_LOS = DICT_POS_LOS[(x1, y1)]
        for point in points_in_LOS:
            img_env[point[0]][point[1]] = dict_of_colors_for_graphics[DARK_RED_N]
        img_env[x1][y1] = dict_of_colors_for_graphics[RED_N]
        for x in range(SIZE_W):
            for y in range(SIZE_H):
                if DSM[x][y] == 1.:
                    img_env[x][y] = dict_of_colors_for_graphics[GREY_N]
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img_env)
        for point in goal_points:
            img_env[point[0]][point[1]] = dict_of_colors_for_graphics[GREEN_N]
        axs[1].imshow(img_env)
        plt.show()
    print("end ", x1, y1)
    return goal_points




def list_dict_2_tuple_dict(in_dict):
    out_dict = {}
    for k,v in in_dict.items():
        if v:
            p = np.array(v)
            out_dict[k] = (p[:,0],p[:,1])
        else:
            out_dict[k] = ([],[])
    return out_dict

def creat_and_save_dictionaries_tuples():
    PREPROCESSING_PATH = os.path.dirname(os.path.realpath(__file__)) #should be in the same dir as all the 'position_los' files
    for filename in os.listdir(PREPROCESSING_PATH):
        if "position_los" in filename and DSM_name in filename:
            outfile = filename[:-4] + "_tuple.pkl"
            print (outfile)
            with open(os.path.join(PREPROCESSING_PATH, filename), 'rb') as f:
                in_dict = pickle.load(f)
            out_dict = list_dict_2_tuple_dict(in_dict)
            with open(os.path.join(PREPROCESSING_PATH, outfile), 'wb') as f_out:
                pickle.dump(out_dict,f_out)


def calc_all_pairs_data(CALC_SHORTEST_PATHS = True):
    import networkx as nx
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

        with open('all_pairs_distances_' + DSM_name + '___' + '.pkl',
                  'wb') as f:
            pickle.dump(all_pairs_distances, f, protocol=2)
            print("finished all_pairs_distances: pickle.dump(all_pairs_distances, f, protocol=2)")

    # print("starting all_pairs_shortest_path")
    # all_pairs_shortest_path = dict(nx.all_pairs_shortest_path(G, cutoff=75))


    if CALC_SHORTEST_PATHS:
        SIZE_W = 100
        SIZE_H = 100
        cutoff = 65
        all_pairs_shortest_path_less_than_65_no_double = {}
        for x1 in range(0, SIZE_W):
            for y1 in range(0, SIZE_H):
                if (x1, y1) not in all_pairs_shortest_path_less_than_65_no_double.keys():
                    print("starting ", str(x1), " ", str(y1))
                    all_pairs_shortest_path_less_than_65_no_double[(x1, y1)] = {}
                else:
                    print("(x1, y1) IS IN all_pairs_shortest_path_less_than_65_no_double.keys()")
                if DSM[x1][y1]==1:
                    continue
                for x2 in range(0, SIZE_W):
                    for y2 in range(0, SIZE_H):
                        if not DSM[x2][y2]==1:
                            if (x1, y1) in all_pairs_distances.keys() and (x2, y2) in all_pairs_distances[(x1, y1)].keys():
                                print(x1, y1, x2, y2)
                                if all_pairs_distances[(x1, y1)][(x2, y2)]<cutoff:
                                    if (x2, y2) in all_pairs_shortest_path_less_than_65_no_double.keys() and (x1, y1) not in all_pairs_shortest_path_less_than_65_no_double[(x2, y2)]:
                                        path = nx.astar_path(G, (x1, y1), (x2, y2))
                                        all_pairs_shortest_path_less_than_65_no_double[(x1, y1)][(x2, y2)] = path

        with open('all_pairs_shortest_path_' + DSM_name + '_' + '65'+ '.pkl',
                  'wb') as f:
            pickle.dump(all_pairs_shortest_path_less_than_65_no_double, f, protocol=2)
            print("finished all_pairs_shortest_path_less_than_65_no_double: pickle.dump(all_pairs_shortest_path_less_than_65_no_double, f, protocol=2)")




if __name__ == '__main__':

    creat_and_save_dictionaries()
    creat_and_save_dictionaries_tuples()

    calc_all_pairs_data(CALC_SHORTEST_PATHS=True)

    # calc_and_save_dominating_points()
    # calc_and_save_lose_points()
    # show_LOS_from_point(5, 5)
    # find_closest_point_not_in_los(5, 5)
    # find_dominating_point(5, 5)
    #find_lose_points(5,5) #{red_pos : points that is blus is in blue will lose!}