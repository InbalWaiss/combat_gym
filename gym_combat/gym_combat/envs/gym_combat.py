import gym
import gym_combat
from gym import error, spaces, utils
from gym.utils import seeding
from gym_combat.envs.Arena.Environment import Environment, Episode
from gym_combat.envs.Greedy import Greedy_player
from gym_combat.envs.Common.constants import *
from gym_combat.envs.Arena.Entity import Entity
from gym_combat.envs.Arena.CState import State


def evaluate(episode_number):
    #if episode_number % EVALUATE_PLAYERS_EVERY == 0:
    a = episode_number % EVALUATE_PLAYERS_EVERY
    if a>=0 and a<EVALUATE_BATCH_SIZE:
        EVALUATE = True
    else:
        EVALUATE = False
    return EVALUATE


class GymCombatEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(SIZE_X, SIZE_Y, 3), dtype=np.uint8)

        self.env = Environment(IS_TRAINING)

        self.red_decision_maker = Greedy_player.Greedy_player()
        self.env.red_player = Entity(self.red_decision_maker)
        self.episode_number = 0


    def reset(self):

        self.episode_number+=1
        EVALUATE = evaluate(self.episode_number)
        self.current_episode = Episode(self.episode_number, EVALUATE=EVALUATE, show_always=False if IS_TRAINING else True)

        # set new start position for the players
        self.env.reset_game(self.episode_number)
        observation_for_blue_s0: State = self.env.get_observation_for_blue()
        return observation_for_blue_s0.img

    def render(self, mode='human', close=False, show = False):
        self.current_episode.print_info_of_episode(self.env, self.current_episode.number_of_steps, 0, self.current_episode.episode_number)

    def step(self, action_blue):
        if NONEDETERMINISTIC_TERMINAL_STATE:

            self.current_episode.number_of_steps += 1
            observation_for_red_s0: State = self.env.get_observation_for_red()
            ##### Blue takes the action #####
            self.env.take_action(Color.Blue, AgentAction(action_blue))  # take the action!

            ##### Red takes the action #####
            action_red: AgentAction = self.red_decision_maker.get_action(observation_for_red_s0)
            self.env.take_action(Color.Red, action_red)  # take the action!

            self.current_episode.is_terminal = (self.env.compute_terminal(whos_turn=Color.Red) is not WinEnum.NoWin)

            if self.current_episode.is_terminal:
                self.env.update_win_counters(self.current_episode.number_of_steps)

            self.current_episode.print_episode(self.env, self.current_episode.number_of_steps)

            observation_for_blue_s1: State = self.env.get_observation_for_blue()

            reward_step_blue, reward_step_red = self.env.handle_reward(self.current_episode.number_of_steps)
            self.current_episode.episode_reward_red += reward_step_red
            self.current_episode.episode_reward_blue += reward_step_blue

            if self.current_episode.number_of_steps == MAX_STEPS_PER_EPISODE:
                # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
                self.current_episode.is_terminal = True

            if self.current_episode.is_terminal:
                self.end_of_episode()

            return observation_for_blue_s1.img, reward_step_blue, self.current_episode.is_terminal, {}


        else: #DETERMINISTIC_TERMINAL_STATE
            self.current_episode.number_of_steps +=1
            ##### Blue takes the action #####
            self.env.take_action(Color.Blue, AgentAction(action_blue))  # take the action!

            self.current_episode.is_terminal = (self.env.compute_terminal(whos_turn=Color.Blue) is not WinEnum.NoWin)
            self.current_episode.print_episode(self.env, self.current_episode.number_of_steps)

            if not self.current_episode.is_terminal:
                ##### Red's turn! #####
                observation_for_red_s0: State = self.env.get_observation_for_red()
                action_red: AgentAction = self.red_decision_maker.get_action(observation_for_red_s0)
                self.env.take_action(Color.Red, action_red)  # take the action!

                # check if red won
                self.current_episode.is_terminal = (self.env.compute_terminal(whos_turn=Color.Red) is not WinEnum.NoWin)

            reward_step_blue, reward_step_red = self.env.handle_reward(self.current_episode.number_of_steps)
            self.current_episode.episode_reward_red += reward_step_red
            self.current_episode.episode_reward_blue += reward_step_blue

            if self.current_episode.is_terminal:
                self.env.update_win_counters(self.current_episode.number_of_steps)

            observation_for_blue_s1: State = self.env.get_observation_for_blue()

            self.current_episode.print_episode(self.env, self.current_episode.number_of_steps)

            if self.current_episode.number_of_steps == MAX_STEPS_PER_EPISODE:
                # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
                self.current_episode.is_terminal = True

            if self.current_episode.is_terminal:
                self.end_of_episode()

            return observation_for_blue_s1.img, reward_step_blue, self.current_episode.is_terminal, {}

    def end_of_episode(self):
        # for statistics
        EVALUATE = self.evaluate()
        self.env.update_win_counters(self.current_episode.number_of_steps)
        self.env.data_for_statistics(self.current_episode.episode_reward_blue, self.current_episode.episode_reward_red, self.current_episode.number_of_steps, 0)
        self.env.evaluate_info(EVALUATE, self.current_episode.episode_number, self.current_episode.number_of_steps, 0)

        if self.current_episode.episode_number % SAVE_STATS_EVERY == 0:
            self.env.end_run()

        # print info of episode:
        #self.current_episode.print_info_of_episode(self.env, self.current_episode.number_of_steps, 0, self.current_episode.episode_number)

    def evaluate(self):
        episode_number = self.current_episode.episode_number
        a = episode_number % EVALUATE_PLAYERS_EVERY
        if a>=0 and a<EVALUATE_BATCH_SIZE:
            EVALUATE = True
        else:
            EVALUATE = False
        return EVALUATE