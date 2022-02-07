import os

import cv2
import matplotlib.pyplot as plt

from gym_combat.gym_combat.envs.Arena import Environment
from gym_combat.gym_combat.envs.Arena.helper_funcs import *
from time import sleep
#from gym_combat.gym_combat.envs.Common.constants import dict_of_colors_for_graphics

buffer_for_win_reward = 5

def print_stats(array_of_results, save_folder_path, plot_every, save_figure=True, steps=False, player=Color.Blue):
    moving_avg = np.convolve(array_of_results, np.ones((plot_every,)) / plot_every, mode='valid')
    plt.figure()
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.xlabel("episode #")
    if steps:
        # plt.axis([0, len(array_of_results), 0, MAX_STEPS_PER_EPISODE])
        plt.axis([0, len(array_of_results), 0, max(moving_avg)])
        plt.suptitle(f"Avg number of steps per episode")
        plt.ylabel(f"steps per episode {SHOW_EVERY}ma")
        if save_figure:
            plt.savefig(save_folder_path + os.path.sep + '#steps_')
            #plt.savefig(save_folder_path + os.path.sep + '#steps_' + str(len(array_of_results) - 1))
    else:
        plt.axis([0, len(array_of_results), LOST_PENALTY - 1, WIN_REWARD + 1])
        #plt.axis([0, len(array_of_results), LOST_PENALTY - 50, WIN_REWARD + 50])
        if player == Color.Blue:
            plt.suptitle(f"Rewards per episode for BLUE player")
        if player == Color.Red:
                plt.suptitle(f"Rewards per episode for RED player")
        plt.ylabel(f"Reward {SHOW_EVERY}ma")
        if save_figure:
            if player == Color.Blue:
                plt.savefig(save_folder_path + os.path.sep + 'rewards_BLUE')
                # plt.savefig(save_folder_path + os.path.sep + 'rewards_BLUE' + str(len(array_of_results) - 1))
            else:
                plt.savefig(save_folder_path + os.path.sep + 'rewards_RED')
                #plt.savefig(save_folder_path + os.path.sep + 'rewards_RED' + str(len(array_of_results) - 1))
    plt.close()
    # plt.show()

def save_reward_stats(save_folder_path, plot_every,  win_array_blue, win_array_red, steps_per_episode, blue_epsilon_values):
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    plt.subplots_adjust(hspace=.4, top=0.9)
    moving_avg_blue = np.convolve(win_array_blue, np.ones((plot_every,)) / plot_every, mode='valid')
    #moving_avg_red = np.convolve(win_array_red, np.ones((plot_every,)) / plot_every, mode='valid')
    # reward_upper_bound = np.max([np.max(moving_avg_blue), np.max(moving_avg_red)])
    # reward_lower_bound = np.min([np.min(moving_avg_blue), np.min(moving_avg_red)])
    reward_upper_bound = np.max(moving_avg_blue)
    reward_lower_bound = np.min(moving_avg_blue)
    # Blue reward:
    axs[0, 0].plot([i for i in range(len(moving_avg_blue))], moving_avg_blue)
    axs[0, 0].set_title(f"Episode rewards BLUE player", fontsize=12, fontweight='bold', color='blue')
    axs[0, 0].axis([0, len(win_array_blue), (reward_lower_bound - buffer_for_win_reward / np.max([reward_lower_bound,1])),
                    (reward_upper_bound + buffer_for_win_reward / np.max([reward_upper_bound,1]))])
    axs[0, 0].set(xlabel="episode #", ylabel=f"Reward {SHOW_EVERY}ma")
    # # Red reward:
    # axs[0, 1].plot([i for i in range(len(moving_avg_red))], moving_avg_red)
    # axs[0, 1].set_title(f"Episode rewards Red player", fontsize=12, fontweight='bold', color='red')
    # axs[0, 1].axis([0, len(win_array_red), (reward_lower_bound - buffer_for_win_reward / np.max([reward_lower_bound,1])),
    #                 int(reward_upper_bound + buffer_for_win_reward / np.max([reward_upper_bound,1]))])
    # axs[0, 1].set(xlabel="episode #", ylabel=f"Reward {SHOW_EVERY}ma")
    # Steps:
    moving_avg = np.convolve(steps_per_episode, np.ones((plot_every,)) / plot_every, mode='valid')
    axs[1, 0].plot([i for i in range(len(moving_avg))], moving_avg)
    axs[1, 0].set_title(f"Avg episode number of steps", fontsize=12, fontweight='bold', color='black')
    axs[1, 0].axis([0, len(steps_per_episode), 0, MAX_STEPS_PER_EPISODE])
    axs[1, 0].set(xlabel="episode #", ylabel=f"steps per episode {SHOW_EVERY}ma")
    # Epsilon:
    moving_avg = np.convolve(blue_epsilon_values, np.ones((plot_every,)) / plot_every, mode='valid')
    axs[1, 1].plot([i for i in range(len(moving_avg))], moving_avg)
    axs[1, 1].set_title(f"Epsilon value per episode", fontsize=12, fontweight='bold', color='black')
    axs[1, 1].axis([0, len(steps_per_episode), -0.1, 1.1])
    axs[1, 1].set(xlabel="episode", ylabel="epsilon")

    plt.savefig(save_folder_path + os.path.sep + 'reward_statistics')
    #plt.savefig(save_folder_path + os.path.sep + 'reward_statistics' + str(len(blue_epsilon_values)))
    plt.close()
    #plt.show()

def save_win_statistics(win_array, blue_epsilon_values, save_folder_path, plot_every):
    win_array = np.array(win_array)
    win_array_blue = (win_array == WinEnum.Blue) * 100
    moving_avg_win_blue = np.convolve(win_array_blue, np.ones((plot_every,)) / plot_every, mode='valid')
    win_array_red = (win_array == WinEnum.Red) * 100
    moving_avg_win_red = np.convolve(win_array_red, np.ones((plot_every,)) / plot_every, mode='valid')
    win_array_NoWin = (win_array == WinEnum.NoWin) * 100
    moving_avg_win_NoWin = np.convolve(win_array_NoWin, np.ones((plot_every,)) / plot_every, mode='valid')
    win_array_Tie = (win_array == WinEnum.Tie) * 100
    moving_avg_win_Tie = np.convolve(win_array_Tie, np.ones((plot_every,)) / plot_every, mode='valid')

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    plt.subplots_adjust(hspace=.4, top=0.9)
    axs[0, 0].plot(moving_avg_win_blue)
    axs[0, 0].set_title('%Blue_win', fontsize=12, fontweight='bold', color='blue')
    axs[0, 0].axis([0, len(moving_avg_win_blue), -5, 105])
    axs[0, 0].set(xlabel="episode")

    axs[0, 1].plot(moving_avg_win_red)
    axs[0, 1].set_title('%Red_win', fontsize=12, fontweight='bold', color='red')
    axs[0, 1].axis([0, len(moving_avg_win_blue), -5, 105])
    axs[0, 1].set(xlabel="episode")


    moving_avg = np.convolve(blue_epsilon_values, np.ones((plot_every,)) / plot_every, mode='valid')
    axs[1, 0].plot([i for i in range(len(moving_avg))], moving_avg)
    axs[1, 0].set_title(f"Epsilon value per episode", fontsize=12, fontweight='bold', color='black')
    axs[1, 0].axis([0, len(moving_avg_win_blue), -0.1, 1.1])
    axs[1, 0].set(xlabel="episode", ylabel="epsilon")

    axs[1, 1].plot(moving_avg_win_Tie)
    axs[1, 1].set_title('%Tie_max_num_steps', fontsize=12, fontweight='bold')
    axs[1, 1].axis([0, len(moving_avg_win_blue), -5, 105])
    axs[1, 1].set(xlabel="episode")
    plt.savefig(save_folder_path + os.path.sep + 'win_statistics')
    #plt.savefig(save_folder_path + os.path.sep + 'win_statistics' + str(len(win_array)))
    plt.close()
    # plt.show()

def save_evaluation_data(evaluation__number_of_steps, evaluation__win_array_blue, evaluation__rewards_for_blue, evaluation__win_array_tie, evaluation__epsilon_value, save_folder_path):
    plot_every = 1
    win_array_blue = np.array(evaluation__win_array_blue)
    moving_avg_win_blue = np.convolve(win_array_blue, np.ones((plot_every,)) / plot_every, mode='valid')
    win_array_Tie = evaluation__win_array_tie
    moving_avg_win_Tie = np.convolve(win_array_Tie, np.ones((plot_every,)) / plot_every, mode='valid')
    moving_avg_win_red = 100-win_array_blue


    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()
    plt.subplots_adjust(hspace=.4, top=0.9)
    axs[0, 0].plot(moving_avg_win_blue)
    axs[0, 0].set_title('%Blue_win', fontsize=10, fontweight='bold', color='blue')
    axs[0, 0].axis([0, len(moving_avg_win_blue), -5, 105])

    axs[0, 1].plot(moving_avg_win_red)
    axs[0, 1].set_title('%Red_win', fontsize=10, fontweight='bold', color='red')
    axs[0, 1].axis([0, len(moving_avg_win_blue), -5, 105])


    moving_avg = np.convolve(evaluation__epsilon_value, np.ones((plot_every,)) / plot_every, mode='valid')
    axs[1, 0].plot([i for i in range(len(moving_avg))], moving_avg)
    axs[1, 0].set_title(f"Epsilon value per episode", fontsize=10, fontweight='bold', color='black')
    axs[1, 0].axis([0, len(moving_avg_win_blue), -0.1, 1.1])

    axs[1, 1].plot(moving_avg_win_Tie)
    axs[1, 1].set_title('%Tie_max_num_steps', fontsize=10, fontweight='bold')
    axs[1, 1].axis([0, len(moving_avg_win_blue), -5, 105])

    # Steps:
    moving_avg = np.convolve(evaluation__number_of_steps, np.ones((plot_every,)) / plot_every, mode='valid')
    axs[0, 2].plot([i for i in range(len(moving_avg))], moving_avg)
    axs[0, 2].set_title(f"Avg number of steps", fontsize=10, fontweight='bold', color='black')
    axs[0, 2].axis([0, len(evaluation__number_of_steps), 0, np.max(evaluation__number_of_steps)+1])
    # blue reward

    moving_avg_blue = np.convolve(evaluation__rewards_for_blue, np.ones((plot_every,)) / plot_every, mode='valid')
    reward_upper_bound = np.max(moving_avg_blue)
    reward_lower_bound = np.min(moving_avg_blue)
    # Blue reward:
    axs[1, 2].plot([i for i in range(len(moving_avg_blue))], moving_avg_blue)
    axs[1, 2].set_title(f"rewards BLUE player", fontsize=10, fontweight='bold', color='blue')
    #axs[1, 2].axis([0, len(win_array_blue), np.min(evaluation__rewards_for_blue)-1, np.max(evaluation__rewards_for_blue)+1 ])
    axs[1, 2].axis([0, len(win_array_blue), (reward_lower_bound - buffer_for_win_reward / np.max([reward_lower_bound,1])),
                    (reward_upper_bound + buffer_for_win_reward / np.max([reward_upper_bound,1]))])


    plt.savefig(save_folder_path + os.path.sep + 'evaluation_statistics')
    #plt.savefig(save_folder_path + os.path.sep + 'evaluation_statistics' + str(len(evaluation__number_of_steps*EVALUATE_PLAYERS_EVERY)))
    plt.close()
    # plt.show()

def print_stats_humna_player(array_of_results, save_folder_path, number_of_episodes, save_figure=True, steps=False,
                             red_player=False):
    moving_avg = np.convolve(array_of_results, np.ones((1,)) / 1, mode='valid')
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.xlabel("episode #")
    if steps:
        plt.axis([0, len(array_of_results), 0, MAX_STEPS_PER_EPISODE])
        if red_player:  # number of steps figure for red player
            plt.suptitle(f"avg number of steps per episode red player")
            plt.ylabel(f"steps{number_of_episodes}")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + '#steps red player')
        else:  # number of steps figure for blue player
            plt.suptitle(f"avg number of steps per episode blue player")
            plt.ylabel(f"steps{number_of_episodes}")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + '#steps blue player')
    else:
        plt.axis([0, len(array_of_results), LOST_PENALTY, WIN_REWARD])
        if red_player:  # reward figure for red player
            plt.suptitle(f"Reward for red player")
            plt.ylabel(f"Rewards")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + 'rewards red player')
        else:  # reward figure for blue player
            plt.suptitle(f"Reward for blue player")
            plt.ylabel(f"Rewards")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + 'rewards blue player')

    plt.show()


PLT_USING_CV2 = False
if PLT_USING_CV2:
    dict_of_colors_for_graphics = dict_of_colors_for_graphics_cv2

def create_image(env: Environment, episode, last_step_number, cover = None):

    blue = env.blue_player
    red = env.red_player
    game_number = episode.episode_number
    number_of_steps = last_step_number
    wins_for_blue = env.wins_for_blue
    wins_for_red = env.wins_for_red
    tie_count = env.tie_count

    # img = img.resize((600, 600))

    if SIZE_H<20:
        const = 30
        margin_h = 2
        margin_w = 0

    else:
        const=5
        margin_h = 8
        margin_w = 0

    informative_env = np.zeros((const * (SIZE_H + margin_h * 2), const * (SIZE_H + margin_w * 2), 3), dtype=np.uint8)

    only_env = np.zeros((const * SIZE_H, const * SIZE_W, 3), dtype=np.uint8)
    for h in range(SIZE_H):
        for w in range(SIZE_W):
            if DSM[h,w] == 1.:
                only_env[h * const: h * const + const, w * const: + w * const + const] = dict_of_colors_for_graphics[GREY_N]

    # add margins to print information on
    informative_env[0:margin_h * const] = dict_of_colors_for_graphics[GREY_N]
    informative_env[(margin_h + SIZE_H) * const:(2 * margin_h + SIZE_H) * const] = dict_of_colors_for_graphics[GREY_N]
    informative_env[margin_h * const: (margin_h + SIZE_H) * const,
    margin_w * const: (margin_w + SIZE_W) * const] = only_env

    radius = int(np.ceil(const / 2))
    thickness = -1

    # Plot Red player
    if env.win_status == WinEnum.Blue:
        red_color_0 = int(np.max([0, dict_of_colors_for_graphics[DARK_DARK_RED_N][0]-30]))
        red_color_1 = int(np.max([0, dict_of_colors_for_graphics[DARK_DARK_RED_N][1]-30]))
        red_color_2 = int(np.max([0, dict_of_colors_for_graphics[DARK_DARK_RED_N][2]-30]))
        points_dom_points_color = (red_color_0, red_color_1, red_color_2)

        los_color_0 = int(np.max([0, dict_of_colors_for_graphics[DARK_RED_N][0]-30]))
        los_color_1 = int(np.max([0, dict_of_colors_for_graphics[DARK_RED_N][1] - 30]))
        los_color_2 = int(np.max([0, dict_of_colors_for_graphics[DARK_RED_N][2] - 30]))
        points_in_enemy_los_color = (los_color_0, los_color_1, los_color_2)

        red_player_color_0 = int(np.max([0, dict_of_colors_for_graphics[RED_N][0]-50]))
        red_player_color_1 = int(np.max([0, dict_of_colors_for_graphics[RED_N][1]-50]))
        red_player_color_2 = int(np.max([0, dict_of_colors_for_graphics[RED_N][2]-50]))
        red_player_color = (red_player_color_0, red_player_color_1, red_player_color_2)
    else: #env.win_status!=WinEnum.Blue
        points_dom_points_color =dict_of_colors_for_graphics[DARK_RED_N]
        points_in_enemy_los_color = dict_of_colors_for_graphics[DARK_DARK_RED_N]
        red_player_color = dict_of_colors_for_graphics[RED_N]

    #if env.win_status!=WinEnum.Blue:

    points_dom_points = DICT_POS_LOS[(red.h, red.w)]
    for point in points_dom_points:
        informative_env[(point[0] + margin_h) * const: (point[0] + margin_h) * const + const,
        (point[1] + margin_w) * const: (point[1] + margin_w) * const + const] = points_in_enemy_los_color


    points_in_enemy_los = DICT_POS_FIRE_RANGE[(red.h, red.w)]
    for point in points_in_enemy_los:
        color = points_dom_points_color
        if NONEDETERMINISTIC_TERMINAL_STATE:
            dist = np.linalg.norm(np.array(point) - np.array([red.h, red.w]))
            dist_floor = np.floor(dist)
            enemy_color = dict_of_colors_for_graphics[RED_N]
            #color = tuple(map(lambda i, j: int(i - j), enemy_color, (0, 0, np.max([10,np.min([points_dom_points_color[2],15 * dist_floor])]))))
            color = tuple(map(lambda i, j: int(i - j), enemy_color,
                              (0, 0, np.max([10, 14 * dist_floor]))))
        informative_env[(point[0] + margin_h) * const: (point[0] + margin_h) * const + const,
        (point[1] + margin_w) * const: (point[1] + margin_w) * const + const] = color

    # set the players as circles
    # set the red player
    center_cord_red_h = (red.h + margin_h) * const + radius
    center_cord_red_w = (red.w + margin_w) * const + radius
    red_color = red_player_color
    cv2.circle(informative_env, (center_cord_red_w, center_cord_red_h), radius, red_color, thickness)

    # Plot Blue player
    center_cord_blue_h = (blue.h + margin_h) * const + radius
    center_cord_blue_w = (blue.w + margin_w) * const + radius
    # plot the blue player
    if env.win_status == WinEnum.Red:
        blue_color_0 = int(np.max([0, dict_of_colors_for_graphics[BLUE_N][0]-150]))
        blue_color_1 = int(np.max([0, dict_of_colors_for_graphics[BLUE_N][1]-150]))
        blue_color_2 = int(np.max([0, dict_of_colors_for_graphics[BLUE_N][2]-150]))
        blue_player_color = (blue_color_0, blue_color_1, blue_color_2)
    else: # env.win_status != WinEnum.Red:
        blue_player_color = dict_of_colors_for_graphics[BLUE_N]


    cv2.circle(informative_env, (center_cord_blue_w, center_cord_blue_h), radius, blue_player_color, thickness)

    # add episode number at the bottom of the window
    font = cv2.FONT_HERSHEY_SIMPLEX
    botoomLeftCornerOfText = (5, (SIZE_W + margin_h * 2) * const - 10)
    fontScale = 0.5
    color = (100, 200, 120)  # greenish
    thickness = 1
    cv2.putText(informative_env, f"episode #{game_number}", botoomLeftCornerOfText, font, 0.7, color, thickness,
                cv2.LINE_AA)


    if env.win_status != WinEnum.NoWin:
        # print who won
        thickness = 2
        botoomLeftCornerOfText_steps = (int(np.floor(SIZE_W / 2)) * const - 79, 55)
        if env.win_status == WinEnum.NoWin:
            botoomLeftCornerOfText = (int(np.floor(SIZE_W / 2)) * const - 38, 30)
            cv2.putText(informative_env, f"No Winner!", botoomLeftCornerOfText, font, fontScale, dict_of_colors_for_graphics[PURPLE_N],
                        thickness, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7,
                        dict_of_colors_for_graphics[PURPLE_N], 0, cv2.LINE_AA)
        elif env.win_status == WinEnum.Red:
            botoomLeftCornerOfText = (int(np.floor(SIZE_W / 2)) * const - 55, 30)
            cv2.putText(informative_env, f"RED WON!", botoomLeftCornerOfText, font, fontScale, dict_of_colors_for_graphics[RED_N],
                        thickness - 1, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7,
                        dict_of_colors_for_graphics[PURPLE_N], 0, cv2.LINE_AA)
        elif env.win_status == WinEnum.Blue:
            botoomLeftCornerOfText = (int(np.floor(SIZE_W / 2)) * const - 50, 30)
            cv2.putText(informative_env, f"BLUE WON!", botoomLeftCornerOfText, font, fontScale, dict_of_colors_for_graphics[BLUE_N],
                        thickness - 1, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7,
                        dict_of_colors_for_graphics[PURPLE_N], 0, cv2.LINE_AA)
        else:  # both lost...
            botoomLeftCornerOfText = (int(np.floor(SIZE_W / 2)) * const - 60, 30)
            cv2.putText(informative_env, f"both lost...", botoomLeftCornerOfText, font, fontScale,
                        dict_of_colors_for_graphics[PURPLE_N], thickness - 1, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7,
                        dict_of_colors_for_graphics[PURPLE_N], 0, cv2.LINE_AA)

       # cv2.waitKey(2)

    else:  # not terminal state
        botoomLeftCornerOfText = (int(np.floor(SIZE_W / 2)) * const - 45, 20)
        cv2.putText(informative_env, f"steps: {number_of_steps}", botoomLeftCornerOfText, font, fontScale,
                    dict_of_colors_for_graphics[PURPLE_N], 0, cv2.LINE_AA)

    # print number of wins
    botoomLeftCornerOfText = (5, 15)
    cv2.putText(informative_env, f"Blue wins: {wins_for_blue}", botoomLeftCornerOfText, font, fontScale,
                dict_of_colors_for_graphics[BLUE_N], 0,
                cv2.LINE_AA)
    botoomLeftCornerOfText = (5, 35)
    cv2.putText(informative_env, f"Red wins: {wins_for_red}", botoomLeftCornerOfText, font, fontScale,
                dict_of_colors_for_graphics[RED_N], 0,
                cv2.LINE_AA)
    botoomLeftCornerOfText = (5, 55)
    cv2.putText(informative_env, f"No Winner : {tie_count}", botoomLeftCornerOfText, font, fontScale,
                dict_of_colors_for_graphics[PURPLE_N], 0,
                cv2.LINE_AA)

    if BB_STATE:
        start_h = np.max([0, blue.h - BB_EXTENSION])
        end_h = np.min([blue.h + BB_EXTENSION + 1, SIZE_H])
        start_w = np.max([0, blue.w - BB_EXTENSION])
        end_w = np.min([blue.w + BB_EXTENSION + 1, SIZE_W])
        informative_env[(start_h + margin_h) * const: (end_h + margin_h) * const + const,
        (start_w + margin_w) * const: (end_w + margin_w) * const + const] += 85
        # set the players as circles
        radius = int(np.ceil(const / 2))
        thickness = -1
        if env.win_status == WinEnum.Blue:
            red_color_0 = int(np.max([0, dict_of_colors_for_graphics[DARK_DARK_RED_N][0] - 30]))
            red_color_1 = int(np.max([0, dict_of_colors_for_graphics[DARK_DARK_RED_N][1] - 30]))
            red_color_2 = int(np.max([0, dict_of_colors_for_graphics[DARK_DARK_RED_N][2] - 30]))
            points_dom_points_color = (red_color_0, red_color_1, red_color_2)

            los_color_0 = int(np.max([0, dict_of_colors_for_graphics[DARK_RED_N][0] - 30-100]))
            los_color_1 = int(np.max([0, dict_of_colors_for_graphics[DARK_RED_N][1] - 30-100]))
            los_color_2 = int(np.max([0, dict_of_colors_for_graphics[DARK_RED_N][2] - 30-100]))
            points_in_enemy_los_color = (los_color_0, los_color_1, los_color_2)

            red_player_color_0 = int(np.max([0, dict_of_colors_for_graphics[RED_N][0] - 50 - 60]))
            red_player_color_1 = int(np.max([0, dict_of_colors_for_graphics[RED_N][1] - 50 - 60]))
            red_player_color_2 = int(np.max([0, dict_of_colors_for_graphics[RED_N][2] - 50 - 60]))
            red_player_color = (red_player_color_0, red_player_color_1, red_player_color_2)
        else:  # env.win_status!=WinEnum.Blue
            points_dom_points_color = dict_of_colors_for_graphics[DARK_RED_N]
            points_in_enemy_los_color = dict_of_colors_for_graphics[DARK_DARK_RED_N]
            red_player_color = dict_of_colors_for_graphics[RED_N]
        # if env.win_status != WinEnum.Blue:
        # set the red player
        #if NONEDETERMINISTIC_TERMINAL_STATE:
        points_dom_points = DICT_POS_LOS[(red.h, red.w)]
        for point in points_dom_points:
            informative_env[(point[0] + margin_h) * const: (point[0] + margin_h) * const + const,
            (point[1] + margin_w) * const: (point[1] + margin_w) * const + const] = points_in_enemy_los_color
        points_in_enemy_fire_range = DICT_POS_FIRE_RANGE[(red.h, red.w)]
        for point in points_in_enemy_fire_range:
            color = points_dom_points_color
            if NONEDETERMINISTIC_TERMINAL_STATE:
                dist = np.linalg.norm(np.array(point) - np.array([red.h, red.w]))
                dist = np.max([0, dist- env.num_steps_red_stay])
                dist_floor = np.floor(dist)
                enemy_color = red_player_color
                #color = tuple(map(lambda i, j: int(i - j), enemy_color, (0, 0, np.max([10,np.min([points_dom_points_color[2],15 * dist_floor])]) )))
                color = tuple(map(lambda i, j: int(i - j), enemy_color,
                                  (0, 0, np.max([10,  14 * dist_floor]))))
            informative_env[(point[0] + margin_h) * const: (point[0] + margin_h) * const + const,
            (point[1] + margin_w) * const: (point[1] + margin_w) * const + const] = color
        # center_cord_red_x = (red.x + margin_x) * const + radius
        # center_cord_red_y = (red.y + margin_y) * const + radius
        # red_color = dict_of_colors_for_graphics[RED_N]
        # cv2.circle(informative_env, (center_cord_red_y, center_cord_red_x), radius, red_color, thickness)
        informative_env[(start_h + margin_h) * const: (end_h + margin_h) * const + const,
        (start_w + margin_w) * const: (end_w + margin_w) * const + const] += 10
        # plot the Red player
        center_cord_red_h = (red.h + margin_h) * const + radius
        center_cord_red_w = (red.w + margin_w) * const + radius
        cv2.circle(informative_env, (center_cord_red_w, center_cord_red_h), radius, red_player_color, thickness)
        # plot the Blue player
        center_cord_blue_h = (blue.h + margin_h) * const + radius
        center_cord_blue_w = (blue.w + margin_w) * const + radius
        cv2.circle(informative_env, (center_cord_blue_w, center_cord_blue_h), radius, blue_player_color, thickness)

        if cover and ((cover[0] != red.h) or (cover[1] != red.w)):
            informative_env[(cover[0] + margin_h) * const: (cover[0] + margin_h) * const + const,
            (cover[1] + margin_w) * const: (cover[1] + margin_w) * const + const] = dict_of_colors_for_graphics[GREEN_N]
    return np.array(informative_env)

def print_episode_graphics(env: Environment, episode, last_step_number, write_file=False, cover = None):
    image = create_image(env, episode, last_step_number, cover)
    if PLT_USING_CV2:
        cv2.imshow("informative_env_"+str(env.combat_env_num), image)  # show it!
        cv2.waitKey(2)
    else:
        plt.title("informative_env_"+str(env.combat_env_num))
        plt.imshow(image)
        plt.pause(.01)
    if episode.is_terminal:
        sleep(1.2)
    # else:
    #     sleep(0.02)