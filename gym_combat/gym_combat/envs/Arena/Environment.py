import time

from gym_combat.gym_combat.envs.Arena.CState import *
from gym_combat.gym_combat.envs.Arena.Position import Position
from gym_combat.gym_combat.envs.Arena.graphics import print_stats, print_episode_graphics, save_win_statistics, \
    save_reward_stats, save_evaluation_data
from gym_combat.gym_combat.envs.Arena.helper_funcs import *
from gym_combat.gym_combat.envs.Common.constants import *
from gym_combat.gym_combat.envs.Arena.Entity import Entity

import numpy as np
from PIL import Image
import pandas as pd
import os


class Environment(object):
    def __init__(self, TRAIN=True, run_name="", combat_env_num=None, move_penalty=-0.01):

        self.blue_player = Entity()
        self.red_player = None

        self.number_of_steps = 0
        self.wins_for_blue = 0
        self.wins_for_red = 0
        self.tie_count = 0
        self.win_status: WinEnum = WinEnum.NoWin
        self.end_game_flag = False

        self.combat_env_num = combat_env_num
        self.move_penalty = move_penalty
        self.enemy_los_penealty = 2 * self.move_penalty

        if TRAIN:
            self.SHOW_EVERY = SHOW_EVERY
            self.NUMBER_OF_EPISODES = NUM_OF_EPISODES
        else:
            self.SHOW_EVERY = EVALUATE_SHOW_EVERY
            self.NUMBER_OF_EPISODES = EVALUATE_NUM_OF_EPISODES
        if run_name != "":
            self.collect_stats = True
            self.create_path_for_statistics(run_name)
        else:
            self.collect_stats = False

        self.num_steps_blue_stay = 0
        self.num_steps_red_stay = 0

        # data for statistics
        self.episodes_rewards_blue_temp = []
        self.episodes_rewards_blue = []
        self.episodes_rewards_blue.append(0)
        self.win_array = []
        self.steps_per_episode_temp = []
        self.steps_per_episode = []
        self.steps_per_episode.append(0)
        self.blue_epsilon_values_temp = []
        self.blue_epsilon_values = []
        self.blue_epsilon_values.append(1)

        # data for evaluation
        self.evaluation__number_of_steps_batch = []
        self.evaluation__win_array_batch = []
        self.evaluation__rewards_for_blue_batch = []
        self.evaluation__epsilon_value_batch = []
        self.evaluation__number_of_steps = []
        self.evaluation__win_array_blue = []
        self.evaluation__win_array_tie = []
        self.evaluation__rewards_for_blue = []
        self.evaluation__epsilon_value = []

    def create_path_for_statistics(self, run_name):
        save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH,
                                     format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
                                            f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}")
                                     + "_" + run_name + "_" + DSM_name)
        if self.combat_env_num:
            save_folder_path = path.join(save_folder_path, str(self.combat_env_num))
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        self.path_for_run = save_folder_path

    def reset_game(self, episode_number):
        self.end_game_flag = False
        self.reset_players_positions(episode_number)
        self.win_status: WinEnum = WinEnum.NoWin

    def reset_players_positions_for_VBS_DEMO(self, episode_number):
        # for VBS_DEMO 13/1/2022 an agent was trained for 100 fixed start points
        from fixed_start_positions_for_VBS_DEMO import start_positions
        index = episode_number % 100
        self.blue_player.h = start_positions[index][0]
        self.blue_player.w = start_positions[index][1]
        self.red_player.h = start_positions[index][2]
        self.red_player.w = start_positions[index][3]

    def reset_players_positions(self, episode_number):
        legal_start_points = False
        while not legal_start_points:
            self.blue_player._choose_random_position()
            if CLOSE_START_POSITION:
                legal_start_points = self._choose_second_position()
            else:
                self.red_player._choose_random_position()
                is_los = (self.red_player.h, self.red_player.w) in DICT_POS_FIRE_RANGE[
                    (self.blue_player.h, self.blue_player.w)]
                legal_start_points = not is_los

    def _choose_second_position(self):
        first_player_w = self.blue_player.w
        first_player_h = self.blue_player.h

        # BB for second flayer start position
        min_cord_w = np.min([np.max([0, first_player_w - 1 * (FIRE_RANGE + BB_MARGIN)]), SIZE_W])
        max_cord_w = np.max([0, np.min([SIZE_W, first_player_w + 1 * (FIRE_RANGE + BB_MARGIN)])])
        min_cord_h = np.min([np.max([0, first_player_h - 1 * (FIRE_RANGE + BB_MARGIN)]), SIZE_H])
        max_cord_h = np.max([0, np.min([SIZE_H, first_player_h + 1 * (FIRE_RANGE + BB_MARGIN)])])

        # find start position for second player: 1) inside BB, 2) not obs, 3) not in LOS of first
        is_obs = True
        while is_obs:
            self.red_player.w = np.random.randint(min_cord_w, max_cord_w)
            self.red_player.h = np.random.randint(min_cord_h, max_cord_h)
            is_obs = self.red_player.is_obs(self.red_player.h, self.red_player.w)

        is_los = (self.red_player.h, self.red_player.w) in DICT_POS_FIRE_RANGE[(self.blue_player.h, self.blue_player.w)]
        if is_los:
            return False

        has_path = False
        if (first_player_h, first_player_w) in all_pairs_distances.keys():
            if (self.red_player.h, self.red_player.w) in all_pairs_distances[(first_player_h, first_player_w)].keys():
                dist = all_pairs_distances[(first_player_h, first_player_w)][(self.red_player.h, self.red_player.w)]
                if dist > MIN_PATH_DIST_FOR_START_POINTS:
                    has_path = True
        elif (np.abs(first_player_h - self.red_player.h) + np.abs(first_player_w - self.red_player.w)) > (
                MIN_PATH_DIST_FOR_START_POINTS):
            has_path = True
        if has_path:
            return True
        return False

    def update_win_counters(self, steps_current_game):
        if steps_current_game == MAX_STEPS_PER_EPISODE:
            self.win_status = WinEnum.Tie
            self.win_array.append(self.win_status)
            self.tie_count += 1
            return

        if not self.end_game_flag:
            return

        if self.win_status == WinEnum.Red:
            self.wins_for_red += 1
            self.win_array.append(WinEnum.Red)
        elif self.win_status == WinEnum.Blue:
            self.wins_for_blue += 1
            self.win_array.append(WinEnum.Blue)
        else:
            print("Bug in update_win_counters")

    def handle_reward(self, steps_current_game):
        if not self.end_game_flag or steps_current_game == MAX_STEPS_PER_EPISODE:
            # reward_step_blue = MOVE_PENALTY
            # reward_step_red = MOVE_PENALTY
            reward_step_blue = self.move_penalty
            reward_step_red = self.move_penalty

            red_pos = self.red_player.get_coordinates()
            blue_pos = self.blue_player.get_coordinates()
            points_in_enemy_los = DICT_POS_LOS[red_pos]
            if blue_pos in points_in_enemy_los:
                # reward_step_blue= ENEMY_LOS_PENALTY
                reward_step_blue = self.enemy_los_penealty

            return reward_step_blue, reward_step_red

        # Game has ended!
        if self.win_status == WinEnum.Red:
            reward_step_blue = LOST_PENALTY
            reward_step_red = WIN_REWARD

        elif self.win_status == WinEnum.Blue:
            reward_step_blue = WIN_REWARD
            reward_step_red = LOST_PENALTY

        else:
            reward_step_blue = 0
            reward_step_red = 0
            print("Bug in handle_reward- WHOS TURN?")

        return reward_step_blue, reward_step_red

    def compute_terminal(self, whos_turn=None) -> WinEnum:
        first_player = self.blue_player
        second_player = self.red_player
        win_status = WinEnum.NoWin

        # sanity check for two players with visibility range
        is_los_first_second = (second_player.h, second_player.w) in DICT_POS_FIRE_RANGE[
            (first_player.h, first_player.w)]
        is_los_second_first = (first_player.h, first_player.w) in DICT_POS_FIRE_RANGE[
            (second_player.h, second_player.w)]
        assert is_los_first_second == is_los_second_first

        is_in_fireRange = (second_player.h, second_player.w) in DICT_POS_FIRE_RANGE[(first_player.h, first_player.w)]
        if not is_in_fireRange:  # not in fire range
            win_status = WinEnum.NoWin
            self.win_status = win_status
            return win_status

        # if SIMULTANEOUS_STEPS: Blue tries, then red tries. needs to be coin toss how's first!
        if whos_turn == Color.Blue or SIMULTANEOUS_STEPS:
            blue_hits = self.blue_player_shoots_and_hits()
            if blue_hits:
                self.end_game_flag = True
                self.win_status = WinEnum.Blue
                return self.win_status

        if whos_turn == Color.Red or SIMULTANEOUS_STEPS:
            red_hits = self.red_player_shoots_and_hits()
            if red_hits:
                self.end_game_flag = True
                self.win_status = WinEnum.Red
                return self.win_status

        self.win_status = win_status
        return win_status

    def blue_player_shoots_and_hits(self):
        # check if blue flayer hit red player
        dist = np.linalg.norm(
            np.array([self.blue_player.h, self.blue_player.w]) - np.array([self.red_player.h, self.red_player.w]))

        if NONEDETERMINISTIC_TERMINAL_STATE:
            dist = np.max([dist, 1])
            p = np.min([(1 / dist) * 3 + (1 / FIRE_RANGE) * self.num_steps_red_stay, 1])
            r = np.random.rand()
            if r <= p:  # blue takes a shoot
                return True
        else:
            if dist <= FIRE_RANGE:
                return True

        return False

    def red_player_shoots_and_hits(self):
        # check if red flayer hit blue player
        dist = np.linalg.norm(
            np.array([self.blue_player.h, self.blue_player.w]) - np.array([self.red_player.h, self.red_player.w]))

        if NONEDETERMINISTIC_TERMINAL_STATE:
            dist = np.max([dist, 1])
            p = np.min([(1 / dist) * 3 + (1 / FIRE_RANGE) * self.num_steps_blue_stay, 1])
            r = np.random.rand()
            if r <= p:  # blue takes a shoot
                return True
        else:
            if dist <= FIRE_RANGE:
                return True

        return False

    def get_observation_for_blue(self) -> State:
        # create observation for blue player
        blue_pos = Position(self.blue_player.h, self.blue_player.w)
        red_pos = Position(self.red_player.h, self.red_player.w)
        if self.win_status == WinEnum.Blue:
            ret_val = State(my_pos=blue_pos, enemy_pos=None)
        elif self.win_status == WinEnum.Red:
            ret_val = State(my_pos=blue_pos, enemy_pos=red_pos, Red_won=True)
        else:
            ret_val = State(my_pos=blue_pos, enemy_pos=red_pos)
            # if we want to count frames in LOS:
            # ret_val = State(my_pos=blue_pos, enemy_pos=red_pos, number_of_steps_blue_stay=self.num_steps_blue_stay, number_of_steps_red_stay=self.num_steps_red_stay, whos_turn=Color.Blue)

        return ret_val

    def get_observation_for_red(self) -> State:
        # create observation for red player
        blue_pos = Position(self.blue_player.h, self.blue_player.w)
        red_pos = Position(self.red_player.h, self.red_player.w)
        if self.win_status == WinEnum.Blue:
            ret_val = State(my_pos=red_pos, enemy_pos=None)
        elif self.win_status == WinEnum.Red:
            ret_val = State(my_pos=red_pos, enemy_pos=blue_pos, Red_won=True)
        else:
            ret_val = State(my_pos=red_pos,
                            enemy_pos=blue_pos)
            # if we want to count frames in LOS:
            # ret_val = State(my_pos=red_pos, enemy_pos=blue_pos, number_of_steps_blue_stay=self.num_steps_blue_stay, number_of_steps_red_stay=self.num_steps_red_stay,  whos_turn=Color.Red)

        return ret_val

    def take_action(self, player_color, action):
        # change the position in world according to player and action
        if self.end_game_flag:
            self.num_steps_blue_stay = 0
            self.num_steps_red_stay = 0
            return action

        if player_color == Color.Red:
            if RED_PLAYER_MOVES:
                self.red_player.action(action)
                if action == AgentAction.Stay:
                    self.num_steps_red_stay += 1
                else:
                    self.num_steps_red_stay = 0
            else:
                self.num_steps_red_stay += 1

        else:  # player_color==Color.Blue
            if TAKE_WINNING_STEP_BLUE and not NONEDETERMINISTIC_TERMINAL_STATE:
                ret_val, winning_action = self.can_blue_win()
                if ret_val:
                    action = winning_action
            self.blue_player.action(action)
            if action == AgentAction.Stay:
                self.num_steps_blue_stay += 1
            else:
                self.num_steps_blue_stay = 0
            return action

    def check_if_blue_and_red_same_pos(self):
        if self.blue_player.h == self.red_player.h:
            if self.blue_player.w == self.red_player.w:
                return True
        return False

    def can_red_win(self):
        # function to check if there is an action that red can take from current state to win
        blue_player = self.blue_player
        red_player = self.red_player
        DEBUG = False

        org_cor_blue_player_h, org_cor_blue_player_w = blue_player.get_coordinates()
        org_cor_red_player_h, org_cor_red_player_w = red_player.get_coordinates()

        ret_val = False
        winning_point_for_red = [-1, -1]
        winning_state = self.get_observation_for_blue()
        winning_action = AgentAction.Stay

        if not RED_PLAYER_MOVES:
            return ret_val, winning_state, winning_action

        for action in range(0, NUMBER_OF_ACTIONS):

            red_player.set_coordinatess(org_cor_red_player_h, org_cor_red_player_w)
            red_player.action(action)
            org_cor_blue_player_h, org_cor_blue_player_w = blue_player.get_coordinates()

            is_los = (org_cor_blue_player_h, org_cor_blue_player_w) in DICT_POS_FIRE_RANGE[
                (red_player.h, red_player.w)]

            if is_los:
                dist = np.linalg.norm(np.array([blue_player.h, blue_player.w]) - np.array([red_player.h, red_player.w]))

                number_of_wins = 0
                for i in range(3):
                    red_won = self.red_player_shoots_and_hits()
                    if red_won:
                        number_of_wins += 1
                if number_of_wins == 3:
                    ret_val = True

                    winning_point_for_red = (red_player.h, red_player.w)
                    blue_pos = Position(blue_player.h, blue_player.w)
                    red_pos = Position(winning_point_for_red[0], winning_point_for_red[1])
                    winning_state = State(my_pos=blue_pos, enemy_pos=red_pos)
                    # Red Takes winning move!!!
                    return ret_val, winning_state, AgentAction(action)

        red_player.set_coordinatess(org_cor_red_player_h, org_cor_red_player_w)
        if DEBUG:
            red_player.set_coordinatess(org_cor_red_player_h, org_cor_red_player_w)
            import matplotlib.pyplot as plt
            blue_obs_satrt = self.get_observation_for_blue()
            plt.matshow(blue_obs_satrt.img)
            plt.show()

            blue_pos = Position(blue_player.h, blue_player.w)
            red_pos = Position(winning_point_for_red[0], winning_point_for_red[1])
            winning_state = State(my_pos=blue_pos, enemy_pos=red_pos)
            plt.matshow(winning_state.img)
            plt.show()

        return ret_val, winning_state, winning_action

    def can_blue_win(self):
        # function to check if there is an action that blue can take from current state to win
        blue_player = self.blue_player
        red_player = self.red_player
        DEBUG = False

        org_cor_blue_player_h, org_cor_blue_player_w = blue_player.get_coordinates()
        org_cor_red_player_h, org_cor_red_player_w = red_player.get_coordinates()

        ret_val = False
        winning_point_for_blue = [-1, -1]
        winning_state = self.get_observation_for_red()
        winning_action = AgentAction.Stay

        for action in range(0, NUMBER_OF_ACTIONS):

            blue_player.set_coordinatess(org_cor_blue_player_h, org_cor_blue_player_w)
            blue_player.action(AgentAction(action))
            org_cor_red_player_h, org_cor_red_player_w = red_player.get_coordinates()

            is_los = (org_cor_red_player_h, org_cor_red_player_w) in DICT_POS_FIRE_RANGE[
                (blue_player.h, blue_player.w)]

            if is_los:
                number_of_wins = 0
                for i in range(3): # numbers of tries for nondeterministic terminal state
                    blue_won = self.blue_player_shoots_and_hits()
                    if blue_won:
                        number_of_wins += 1
                if number_of_wins == 3:
                    ret_val = True

                    # for DEBUG:
                    # winning_point_for_blue = (blue_player.x, blue_player.y)
                    # red_pos = Position(red_player.x, red_player.y)
                    # blue_pos = Position(winning_point_for_blue[0], winning_point_for_blue[1])
                    # winning_state = State(my_pos=red_pos, enemy_pos=blue_pos)

                    blue_player.set_coordinatess(org_cor_blue_player_h, org_cor_blue_player_w)
                    return ret_val, AgentAction(action)

        blue_player.set_coordinatess(org_cor_blue_player_h, org_cor_blue_player_w)
        if DEBUG:
            red_player.set_coordinatess(org_cor_red_player_h, org_cor_red_player_w)
            import matplotlib.pyplot as plt
            red_obs_satrt = self.get_observation_for_red()
            plt.matshow(red_obs_satrt.img)
            plt.show()

            red_pos = Position(red_player.h, red_player.w)
            blue_pos = Position(winning_point_for_blue[0], winning_point_for_blue[1])
            winning_state = State(my_pos=blue_pos, enemy_pos=red_pos)
            plt.matshow(winning_state.img)
            plt.show()

        return ret_val, winning_action

    def evaluate_info(self, EVALUATE_FLAG, episode_number, steps_current_game, blue_epsilon):
        # save statistics
        if EVALUATE_FLAG:
            self.evaluation__number_of_steps_batch.append(steps_current_game)
            self.evaluation__win_array_batch.append(self.win_status)
            self.evaluation__rewards_for_blue_batch.append(self.episodes_rewards_blue[-1])
            self.evaluation__epsilon_value_batch.append(blue_epsilon)

        if episode_number % EVALUATE_PLAYERS_EVERY == EVALUATE_BATCH_SIZE:
            self.evaluation__number_of_steps.append(np.mean(self.evaluation__number_of_steps_batch))
            self.evaluation__rewards_for_blue.append(np.mean(self.evaluation__rewards_for_blue_batch))
            self.evaluation__epsilon_value.append(np.mean(self.evaluation__epsilon_value_batch))

            win_array = np.array(self.evaluation__win_array_batch)
            win_array_blue = (win_array == WinEnum.Blue) * 100
            self.evaluation__win_array_blue.append(np.mean(win_array_blue))

            win_array_Tie = (win_array == WinEnum.Tie) * 100
            self.evaluation__win_array_tie.append(np.mean(win_array_Tie))

            print("\nEvaluation summary - env " + str(self.combat_env_num) + ":",
                  len(self.evaluation__number_of_steps_batch)
                  , "episodes ends in episode number", episode_number, ", epsilon is: ",
                  np.mean(self.evaluation__epsilon_value_batch))
            print("Avg number of steps: ", np.mean(self.evaluation__number_of_steps_batch))
            print("Avg reward for Blue: ", np.mean(self.evaluation__rewards_for_blue_batch))
            print("Win % for Blue: ", np.mean(win_array_blue))

            self.evaluation__number_of_steps_batch = []
            self.evaluation__win_array_batch = []
            self.evaluation__rewards_for_blue_batch = []
            self.evaluation__epsilon_value_batch = []

        return

    def end_run(self):
        if self.collect_stats == False:
            return
        # STATS_RESULTS_RELATIVE_PATH_THIS_RUN = os.path.join(self.path_for_run, STATS_RESULTS_RELATIVE_PATH)
        # self.save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH_THIS_RUN,
        #                              format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
        #                                    f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}_{str(STR_FOLDER_NAME)}"))
        self.save_folder_path = self.path_for_run
        # save info on run
        self.save_stats(self.save_folder_path)

        # print and save figures
        print_stats(self.episodes_rewards_blue, self.save_folder_path, self.SHOW_EVERY, player=Color.Blue)
        print_stats(self.steps_per_episode, self.save_folder_path, self.SHOW_EVERY, save_figure=True, steps=True,
                    player=Color.Blue)

        save_reward_stats(self.save_folder_path, self.SHOW_EVERY, self.episodes_rewards_blue, [],
                          self.steps_per_episode, self.blue_epsilon_values)

        save_win_statistics(self.win_array, self.blue_epsilon_values, self.save_folder_path, self.SHOW_EVERY)

        save_evaluation_data(self.evaluation__number_of_steps, self.evaluation__win_array_blue,
                             self.evaluation__rewards_for_blue, self.evaluation__win_array_tie,
                             self.evaluation__epsilon_value, self.save_folder_path)

    def data_for_statistics(self, episode_reward_blue, episode_reward_red, steps_current_game, blue_epsilon):

        self.episodes_rewards_blue.append(episode_reward_blue)
        self.steps_per_episode.append(steps_current_game)
        self.blue_epsilon_values.append(blue_epsilon)

    def save_stats(self, save_folder_path):

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        info = {f"NUM_OF_EPISODES": [NUM_OF_EPISODES],
                f"MOVE_PENALTY": [self.move_penalty],
                f"WIN_REWARD": [WIN_REWARD],
                f"LOST_PENALTY": [LOST_PENALTY],
                f"ENEMY_LOS_PENALTY": [self.enemy_los_penealty],
                f"ACTION_SPACE_9": [ACTION_SPACE_9],
                f"FIRE_RANGE": [FIRE_RANGE],
                f"DSM_NAME": [DSM_name],
                f"RED_PLAYER_MOVES": [RED_PLAYER_MOVES],
                f"FIXED_START_POINT_RED": [FIXED_START_POINT_RED],
                f"FIXED_START_POINT_BLUE": [FIXED_START_POINT_BLUE],
                f"%WINS_BLUE": [self.wins_for_blue / self.NUMBER_OF_EPISODES * 100],
                f"%WINS_RED": [self.wins_for_red / self.NUMBER_OF_EPISODES * 100],
                f"%TIES": [self.tie_count / self.NUMBER_OF_EPISODES * 100],
                }

        df = pd.DataFrame(info)
        df.to_csv(os.path.join(save_folder_path, 'Statistics.csv'), index=False)

        # save models
        self.red_player._decision_maker.save_model(self.episodes_rewards_blue, save_folder_path, Color.Red)
        if not BASELINES_RUN:
            self.blue_player._decision_maker.save_model(self.episodes_rewards_blue, save_folder_path, Color.Red)


class Episode():
    def __init__(self, episode_number, EVALUATE=False):
        self.episode_number = episode_number
        self.episode_reward_blue = 0
        self.episode_reward_red = 0
        self.number_of_steps = 0
        self.is_terminal = False
        self.show = self.show_episode(EVALUATE)

    def update_number_of_steps(self):
        self.number_of_steps += 1

    def show_episode(self, EVALUATE):
        if EVALUATE or self.episode_number == 1:
            return True
        return False

    def print_episode(self, env, last_step_number, save_file=False, cover=None):
        if self.show and USE_DISPLAY:
            print_episode_graphics(env, self, last_step_number, save_file, cover)

    def print_info_of_episode(self, env, steps_current_game, blue_epsilon, episode_number):
        if self.show:
            if len(env.episodes_rewards_blue) < env.SHOW_EVERY:
                number_of_episodes = len(env.episodes_rewards_blue[-env.SHOW_EVERY:]) - 1
            else:
                number_of_episodes = env.SHOW_EVERY

            if not BASELINES_RUN:
                print(f"\non #{self.episode_number}:")

                print(f"reward for blue player is: , {self.episode_reward_blue}")
                print(f"epsilon (blue player) is {blue_epsilon}")
                print(f"number of steps: {steps_current_game}")

            self.print_episode(env, steps_current_game)

        if self.episode_number % SAVE_STATS_EVERY == 0:
            env.end_run()
