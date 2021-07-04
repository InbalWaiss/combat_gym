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
DSM_names = {"15X15", "100X100_Berlin", "100X100_Paris", "100X100_Boston"}
DSM_name = "100X100_Berlin"


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
    SIZE_X = 15
    SIZE_Y = 15
    FIRE_RANGE = 7
    MAX_STEPS_PER_EPISODE = 100
    BB_STATE = False
    CLOSE_START_POSITION = False
    LOS_PENALTY_RANGE = 2 * FIRE_RANGE
    BB_MARGIN = 0
    SIZE_X_BB = SIZE_X
    SIZE_Y_BB = SIZE_Y
    MIN_PATH_DIST_FOR_START_POINTS = 2
    all_pairs_distances_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '___' + '.pkl'
    if path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")

elif DSM_name=="100X100_Berlin":

    #DSM = get_DSM_berlin()
    #np.savetxt("gym_combat/gym_combat/envs/Common/maps/Berlin/Berlin_1_256_inbal.txt", DSM, fmt="%d")

    SIZE_X=100
    SIZE_Y=100
    DSM = np.loadtxt("gym_combat/gym_combat/envs/Common/maps/Berlin_1_256_inbal.txt", usecols=range(SIZE_X))
    if False:
        import matplotlib.pyplot as plt
        plt.matshow(DSM)
        plt.show()

    FIRE_RANGE = 10
    LOS_PENALTY_RANGE = 3 * FIRE_RANGE
    MAX_STEPS_PER_EPISODE = 250
    MIN_PATH_DIST_FOR_START_POINTS = 2
    BB_STATE = True
    if BASELINES_RUN:
        BB_MARGIN = 5
    else:
        BB_MARGIN = 3
    SIZE_X_BB = 2 * FIRE_RANGE + 2 * BB_MARGIN + 1
    SIZE_Y_BB = 2 * FIRE_RANGE + 2 * BB_MARGIN + 1
    all_pairs_distances_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '___' + '.pkl'
    if path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")
    SAVE_BERLIN_FIXED_STATE = False

elif DSM_name=="100X100_Paris":
    DSM = get_DSM_Paris()
    SIZE_X=100
    SIZE_Y=100
    FIRE_RANGE = 10
    MAX_STEPS_PER_EPISODE = 250
    BB_STATE = True
    BB_MARGIN = 3
    SIZE_X_BB = 2 * FIRE_RANGE + 2 * BB_MARGIN + 1
    SIZE_Y_BB = 2 * FIRE_RANGE + 2 * BB_MARGIN + 1

elif DSM_name=="100X100_Boston":
    DSM = get_DSM_Boston()
    SIZE_X=100
    SIZE_Y=100
    FIRE_RANGE = 10
    MAX_STEPS_PER_EPISODE = 250
    BB_STATE = True
    BB_MARGIN = 3
    SIZE_X_BB = 2 * FIRE_RANGE + 2 * BB_MARGIN + 1
    SIZE_Y_BB = 2 * FIRE_RANGE + 2 * BB_MARGIN + 1
if False:
    import matplotlib.pyplot as plt
    plt.matshow(DSM)
    plt.show()


try:
    with open('gym_combat/gym_combat/envs/Common/Preprocessing/dictionary_position_los_'+DSM_name+'_'+str(FIRE_RANGE)+ '.pkl', 'rb') as f:
        DICT_POS_FIRE_RANGE = pickle.load(f)
except:
    try:
        with open('dictionary_position_los_'+DSM_name+'_'+str(FIRE_RANGE)+'.pkl', 'rb') as f:
            DICT_POS_FIRE_RANGE = pickle.load(f)
    except:
        try:
            with open('/Common/Preprocessing/dictionary_position_los_'+DSM_name+'_'+str(FIRE_RANGE)+'.pkl', 'rb') as f:
                DICT_POS_FIRE_RANGE = pickle.load(f)
        except:
            pass

try:
    with open('gym_combat/gym_combat/envs/Common/Preprocessing/dictionary_position_los_' + DSM_name+ '_'+str(LOS_PENALTY_RANGE)+ '.pkl', 'rb') as f:
        DICT_POS_LOS = pickle.load(f)
except:
    try:
        with open('dictionary_position_los_' +DSM_name+'_'+str(LOS_PENALTY_RANGE)+ '.pkl', 'rb') as f:
            DICT_POS_LOS = pickle.load(f)
    except:
        try:
            with open('../Common/Preprocessing/dictionary_position_los_' +DSM_name+ '_'+str(LOS_PENALTY_RANGE)+'.pkl', 'rb') as f:
                DICT_POS_LOS = pickle.load(f)
        except:
            pass



if BASELINES_RUN:
    MOVE_PENALTY = -0.1
    WIN_REWARD = 3
    LOST_PENALTY = -3
    ENEMY_LOS_PENALTY = MOVE_PENALTY * 2
    TIE = 0

else:
    MOVE_PENALTY = -0.05
    WIN_REWARD = 1
    LOST_PENALTY = -1
    ENEMY_LOS_PENALTY = MOVE_PENALTY*2
    TIE = 0




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

class WinEnum(IntEnum):

    Blue = 0
    Red = 1
    Tie = 2
    NoWin = 3

USE_OLD_COLORS = False
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


dict_of_colors_for_graphics = {1: (239, 0, 0),  #blue
                               2: (175, 0, 0),  #darker blue
                               3: (0, 0, 239),  # red
                               4: (0, 20, 175),  #dark red
                               5: (230, 100, 150),  #purple
                               6: (60, 255, 255),  #yellow
                               7: (100, 100, 100),  #grey
                               8: (0, 239, 0),  #green
                               9: (0, 0, 0),  #black
                               10: (0, 0, 100),  #bright red
                               11: (0, 0, 50),  #bright bright red
                               12: (0, 0, 75), #dark dark red
                               }

OBSTACLE = 1.

if USE_OLD_COLORS:
    dict_of_colors_for_state = dict_of_colors_for_graphics



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


else: # ACTION_SPACE = 4
    NUMBER_OF_ACTIONS = 4
    class AgentAction(IntEnum):
        Right = 0
        Bottom = 1
        Top = 2
        Left = 3


class AgentType(IntEnum):
    Q_table = 1
    DQN_basic = 2
    DQN_keras = 3
    DQN_temporalAttention = 4
    DQNAgent_spatioalAttention = 5
    Greedy = 6

Agent_type_str = {AgentType.Q_table : "Q_table",
                  AgentType.DQN_basic : "DQN_basic",
                  AgentType.DQN_keras : "DQN_keras",
                  AgentType.DQN_temporalAttention : "DQN_temporalAttention",
                  AgentType.DQNAgent_spatioalAttention : "DQNAgent_spatioalAttention",
                  AgentType.Greedy : "Greedy_player"}

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
USE_DISPLAY = False
SHOW_EVERY =50
NUM_OF_EPISODES = 3_000_000+EVALUATE_BATCH_SIZE
SAVE_STATS_EVERY = 10000+EVALUATE_BATCH_SIZE

# training mode
IS_TRAINING = True

