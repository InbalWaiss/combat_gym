from enum import IntEnum
from PIL import Image
import numpy as np
from os import path
import pickle
from gym_combat.gym_combat.envs.Common.Preprocessing.load_DSM_from_excel import get_DSM_berlin, get_DSM_Boston, get_DSM_Paris

BASELINES_RUN = True
SAVE_BERLIN_FIXED_STATE = False

ACTION_SPACE_9 = True
ACTION_SPACE_4 = False
if not ACTION_SPACE_9:
    ACTION_SPACE_4 = True

#RED_TYPE = 'Greedy'
RED_TYPE = 'Smart'

RED_PLAYER_MOVES = True
FIXED_START_POINT_RED = False
FIXED_START_POINT_BLUE = False
TAKE_WINNING_STEP_BLUE = False

NONEDETERMINISTIC_TERMINAL_STATE = False
SIMULTANEOUS_STEPS = False
if SIMULTANEOUS_STEPS:
    NONEDETERMINISTIC_TERMINAL_STATE = True

#image state mode
CLOSE_START_POSITION = True

FULLY_CONNECTED = False
NUM_FRAMES = 1
STR_FOLDER_NAME = "main_berlin_cnn" #"NONEDETERMINISTIC_SIMULTANEOUS_15X15"

#1 is an obstacle
DSM_names = {"15X15", "100X100_Berlin", "100X100_Paris", "100X100_Boston", "Baqa"}
#DSM_name = "100X100_Berlin"
DSM_name = "Baqa"


COMMON_PATH = path.dirname(path.realpath(__file__))
MAIN_PATH = path.dirname(COMMON_PATH)
OUTPUT_DIR = path.join(MAIN_PATH, 'Arena')
STATS_RESULTS_RELATIVE_PATH = "statistics"
RELATIVE_PATH_HUMAN_VS_MACHINE_DATA = path.join(MAIN_PATH, 'gym_combat/gym_combat/envsQtable/trained_agents')


if DSM_name=="15X15":
    DSM = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ])
    SIZE_W = 15
    SIZE_H = 15
    FIRE_RANGE = 7
    MAX_STEPS_PER_EPISODE = 250 # 100
    BB_STATE = False
    CLOSE_START_POSITION = False
    LOS_PENALTY_RANGE = 2 * FIRE_RANGE
    BB_MARGIN = 0
    SIZE_W_BB = SIZE_W
    SIZE_H_BB = SIZE_H
    MIN_PATH_DIST_FOR_START_POINTS = 2
    all_pairs_distances_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '___' + '.pkl'
    if path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")

elif DSM_name=="100X100_Berlin":

    #DSM = get_DSM_berlin()
    #np.savetxt("gym_combat/gym_combat/envs/Common/maps/Berlin/Berlin_1_256_inbal.txt", DSM, fmt="%d")

    SIZE_H=100
    SIZE_W=100
    DSM = np.loadtxt("gym_combat/gym_combat/envs/Common/maps/Berlin_1_256.txt", dtype = np.uint8, usecols=range(SIZE_W))
    if False:
        import matplotlib.pyplot as plt
        plt.matshow(DSM)
        plt.show()

    FIRE_RANGE = 10
    LOS_PENALTY_RANGE = 3 * FIRE_RANGE
    MAX_STEPS_PER_EPISODE = 100
    MIN_PATH_DIST_FOR_START_POINTS = 2
    BB_STATE = True
    if BASELINES_RUN:
        BB_MARGIN = 16
    else:
        BB_MARGIN = 3
    SIZE_W_BB = 4 * FIRE_RANGE + 2 * BB_MARGIN + 1
    SIZE_H_BB = 4 * FIRE_RANGE + 2 * BB_MARGIN + 1
    all_pairs_distances_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '___' + '.pkl'
    if path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")
    all_pairs_shortest_path_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_shortest_path_100X100_Berlin_10.pkl'
    if path.exists(all_pairs_shortest_path_path):
        with open(all_pairs_shortest_path_path, 'rb') as f:
            all_pairs_shortest_path = pickle.load(f)
            print("all_pairs_shortest_path loaded")
    SAVE_BERLIN_FIXED_STATE = False

elif DSM_name=="Baqa":
    SIZE_H = 100
    SIZE_W = 100
    DSM = np.loadtxt(path.join(COMMON_PATH, 'maps', 'BaqaObs.txt'), dtype=np.uint8, usecols=range(SIZE_W))
    if False:
        import matplotlib.pyplot as plt

        plt.matshow(DSM)
        plt.show()

    FIRE_RANGE = 10
    LOS_PENALTY_RANGE = 3 * FIRE_RANGE
    MAX_STEPS_PER_EPISODE =150
    MIN_PATH_DIST_FOR_START_POINTS = 2
    BB_STATE = True
    if BASELINES_RUN:
        BB_MARGIN = 16
    else:
        BB_MARGIN = 3
    SIZE_W_BB = 4 * FIRE_RANGE + 2 * BB_MARGIN + 1
    SIZE_H_BB = 4 * FIRE_RANGE + 2 * BB_MARGIN + 1
    all_pairs_distances_path = path.join(COMMON_PATH, '..', 'Greedy',
                                         'all_pairs_distances_' + DSM_name + '___' + '.pkl')
    #all_pairs_distances_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '___' + '.pkl'
    if path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")

    all_pairs_distances_path_np = path.join(COMMON_PATH, '..', 'Greedy',
                                         'all_pairs_distances_' + DSM_name + 'np' + '.pkl')
    #all_pairs_distances_path_np = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + 'np' + '.pkl'
    if path.exists(all_pairs_distances_path_np):
        with open(all_pairs_distances_path_np, 'rb') as f:
            all_pairs_distances_np = pickle.load(f)
            print("all_pairs_distances_np loaded")

    all_pairs_shortest_path_path = path.join(COMMON_PATH, '..', 'Greedy',
                                         'all_pairs_shortest_pathBaqa_15.pkl')
    #all_pairs_shortest_path_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_shortest_pathBaqa_15.pkl'
    if path.exists(all_pairs_shortest_path_path):
        with open(all_pairs_shortest_path_path, 'rb') as f:
            all_pairs_shortest_path = pickle.load(f)
            print("all_pairs_shortest_path loaded")
    SAVE_BERLIN_FIXED_STATE = False



try:
    DICT_POS_FIRE_RANGE_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_' + DSM_name + '_' + str(FIRE_RANGE) + '.pkl')
    with open(DICT_POS_FIRE_RANGE_path, 'rb') as f:
        DICT_POS_FIRE_RANGE = pickle.load(f)
except:
    print("did not load DICT_POS_FIRE_RANGE")

try:
    DICT_POS_FIRE_RANGE_tuple_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_'+DSM_name+'_'+str(FIRE_RANGE)+ '_tuple.pkl')
    with open(DICT_POS_FIRE_RANGE_tuple_path, 'rb') as f:
        DICT_POS_FIRE_RANGE_TUPLE = pickle.load(f)
except:
    print("did not load DICT_POS_FIRE_RANGE_tuple")

try:
    #turning DICT_POS_FIRE_RANGE to hold sets
    for k in DICT_POS_FIRE_RANGE:
        DICT_POS_FIRE_RANGE[k] = set(DICT_POS_FIRE_RANGE[k])
except:
    print("error in 'turning DICT_POS_FIRE_RANGE to hold sets'")


try:
    DICT_POS_LOS_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_' + DSM_name + '_' + str(
            LOS_PENALTY_RANGE) + '.pkl')
    with open(DICT_POS_LOS_path, 'rb') as f:
        DICT_POS_LOS = pickle.load(f)
except:
    print("did not load DICT_POS_LOS")

try:
    DICT_POS_LOS_TUPLE_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_' + DSM_name+ '_'+str(LOS_PENALTY_RANGE)+ '_tuple.pkl')
    with open(DICT_POS_LOS_TUPLE_path, 'rb') as f:
        DICT_POS_LOS_TUPLE = pickle.load(f)
except:
    print("did not load DICT_POS_LOS_TUPLE")


if BASELINES_RUN:
    MOVE_PENALTY = -0.01
    WIN_REWARD = 1
    LOST_PENALTY = -1
    LOST_PENALTY = 0.5
    ENEMY_LOS_PENALTY = MOVE_PENALTY*2
    TIE = 0

else:
    MOVE_PENALTY = -0.01
    WIN_REWARD = 1
    LOST_PENALTY = -1
    LOST_PENALTY = 0.5
    ENEMY_LOS_PENALTY = MOVE_PENALTY*2
    TIE = 0


class WinEnum(IntEnum):
    Blue = 0
    Red = 1
    Tie = 2
    NoWin = 3

NUMBER_OF_ACTIONS = 9

BLUE_N = 1 #blue player key in dict
DARK_BLUE_N = 2
RED_N = 3 #red player key in dict
DARK_RED_N = 4
DARK_DARK_RED_N = 12
PURPLE_N = 5
YELLOW_N = 6 #to be used for line from blue to red
GREY_N = 7 #obstacle key in dict
GREEN_N = 8
BLACK_N = 9
BRIGHT_RED = 10
BRIGHT_BRIGHT_RED = 11


# for better separation of colors
dict_of_colors_for_state = {1: (0, 0, 255),  #blue
                  2: (0, 0, 175), #darker blue
                  3: (255, 0, 0), # red
                  4: (175, 0, 0), #dark red
                  5: (230, 100, 150), #purple
                  6: (60, 255, 255), #yellow
                  7: (100, 100, 100),#grey
                  8: (0, 255, 0),#green
                  9: (0, 0, 0), #black
                  10: (0, 0, 75), #bright red
                  11: (0, 0, 25), #bright bright red
                  12: (50, 0, 0), #dark dark red
                  }

dict_of_colors_for_graphics = {1: (0, 0, 239),  #blue
                               2: (0, 0, 175),  #darker blue
                               3: (239, 0, 0),  # red
                               4: (175, 20, 0),  #dark red
                               5: (150, 100, 230),  #purple
                               6: (255, 255, 60),  #yellow
                               7: (100, 100, 100),  #grey
                               8: (0, 239, 0),  #green
                               9: (0, 0, 0),  #black
                               10: (100, 0, 0),  #bright red
                               11: (50, 0, 0),  #bright bright red
                               12: (75, 0, 0), #dark dark red
                               }

OBSTACLE = 1.



if ACTION_SPACE_9:
    NUMBER_OF_ACTIONS = 9
    class AgentAction(IntEnum):
        TopRight = 1
        Right = 2
        BottomRight = 3
        Bottom = 4
        Stay = 5
        Top = 6
        BottomLeft = 7
        Left = 8
        TopLeft = 0


class AgentType(IntEnum):
    Q_table = 1
    DQN_basic = 2
    DQN_keras = 3
    DQN_temporalAttention = 4
    DQNAgent_spatioalAttention = 5
    Greedy = 6
    Smart = 7

Agent_type_str = {AgentType.Q_table : "Q_table",
                  AgentType.DQN_basic : "DQN_basic",
                  AgentType.DQN_keras : "DQN_keras",
                  AgentType.DQN_temporalAttention : "DQN_temporalAttention",
                  AgentType.DQNAgent_spatioalAttention : "DQNAgent_spatioalAttention",
                  AgentType.Greedy : "Greedy_player",
                  AgentType.Smart : "smart_player"}

class Color(IntEnum):
    Blue = 1
    Red = 2

# params to evaluate trained models
EVALUATE_SHOW_EVERY = 1
EVALUATE_NUM_OF_EPISODES = 100
EVALUATE_SAVE_STATS_EVERY = 1000

EVALUATE_PLAYERS_EVERY = 1000
EVALUATE_BATCH_SIZE=100

#save information
USE_DISPLAY = True #
SHOW_EVERY = 50
NUM_OF_EPISODES = 3_000_000+EVALUATE_BATCH_SIZE
SAVE_STATS_EVERY = 10000+EVALUATE_BATCH_SIZE

# training mode
IS_TRAINING = True

