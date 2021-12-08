import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_combat.gym_combat.envs.Arena.Environment import Environment, Episode
from gym_combat.gym_combat.envs.Greedy import Greedy_player
from gym_combat.gym_combat.envs.Greedy import smart_player
from gym_combat.gym_combat.envs.Common.constants import *
from gym_combat.gym_combat.envs.Arena.Entity import Entity
from gym_combat.gym_combat.envs.Arena.CState import State
from gym_combat.gym_combat.envs.Arena.graphics import create_image



class GymCombatEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, run_name="", env_num = None, train_mode = True, move_penalty = -0.1):
        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        # Example for using image as input:
        if BB_STATE:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(SIZE_H_BB, SIZE_W_BB, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(SIZE_H, SIZE_W, 3), dtype=np.uint8)
        self.env_num = env_num.get() if env_num else None
        self.env = Environment(IS_TRAINING, run_name, combat_env_num=self.env_num, move_penalty = move_penalty)


        if RED_TYPE == AgentType.Smart:
            self.red_decision_maker = smart_player.SmartPlayer()
        else: # AgentType.Greedy
            self.red_decision_maker = Greedy_player.Greedy_player()

        self.env.red_player = Entity(self.red_decision_maker)

        self.episode_number = 0
        self.train_mode = train_mode



    def reset(self):

        self.episode_number+=1
        EVALUATE = self.evaluate()

        self.current_episode = Episode(self.episode_number, EVALUATE=EVALUATE)

        # set new start position for the players
        self.env.reset_game(self.episode_number)
        self.red_decision_maker.reset_cover()
        observation_for_blue_s0: State = self.env.get_observation_for_blue()
        return observation_for_blue_s0.img

    def render(self, mode='human', close=False, show = False):
        return create_image(self.env, self.current_episode, self.current_episode.number_of_steps, cover = self.red_decision_maker.get_cover())#[...,::-1]
        #self.current_episode.print_info_of_episode(self.env, self.current_episode.number_of_steps, 0, self.current_episode.episode_number)


    def step(self, action_blue):

        if SIMULTANEOUS_STEPS:
            self.current_episode.update_number_of_steps()

            observation_for_red_s0: State = self.env.get_observation_for_red()
            ##### Blue takes the action #####
            self.env.take_action(Color.Blue, AgentAction(action_blue))  # take the action!

            if not self.env.check_if_blue_and_red_same_pos():
                ##### Red takes the action #####
                action_red: AgentAction = self.red_decision_maker.get_action(observation_for_red_s0)
                self.env.take_action(Color.Red, action_red)  # take the action!

            self.current_episode.is_terminal = (self.env.compute_terminal(whos_turn=Color.Red) is not WinEnum.NoWin)

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

            return observation_for_blue_s1.img, reward_step_blue, self.current_episode.is_terminal, {'win':self.env.win_status}

        else: # STEP BY STEP
            self.current_episode.update_number_of_steps()

            ##### Blue takes the action #####
            self.env.take_action(Color.Blue, AgentAction(action_blue))

            self.current_episode.is_terminal = (self.env.compute_terminal(whos_turn=Color.Blue) is not WinEnum.NoWin)
            self.current_episode.print_episode(self.env, self.current_episode.number_of_steps,
                                               cover=self.red_decision_maker.get_cover())

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


            observation_for_blue_s1: State = self.env.get_observation_for_blue()
            self.current_episode.print_episode(self.env, self.current_episode.number_of_steps,
                                               cover=self.red_decision_maker.get_cover())

            if self.current_episode.number_of_steps == MAX_STEPS_PER_EPISODE:
                # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
                self.current_episode.is_terminal = True

            if self.current_episode.is_terminal: #Eran: moved the check of self.train_mode after update_win_counters
                self.end_of_episode()

            return observation_for_blue_s1.img, reward_step_blue, self.current_episode.is_terminal, {'win':self.env.win_status}


    def end_of_episode(self):
        self.env.update_win_counters(self.current_episode.number_of_steps)

        if not self.train_mode:
            return

        # for statistics
        EVALUATE = self.evaluate()
        self.env.data_for_statistics(self.current_episode.episode_reward_blue, self.current_episode.episode_reward_red, self.current_episode.number_of_steps, 0)
        self.env.evaluate_info(EVALUATE, self.current_episode.episode_number, self.current_episode.number_of_steps, 0)

        if self.current_episode.episode_number % SAVE_STATS_EVERY == 0:
            self.env.end_run()

        # print info of episode:
        self.current_episode.print_info_of_episode(self.env, self.current_episode.number_of_steps, 0, self.current_episode.episode_number)

    def evaluate(self):
        if not IS_TRAINING:
            return True

        episode_number = self.episode_number
        a = episode_number % EVALUATE_PLAYERS_EVERY
        if a>0 and a<=EVALUATE_BATCH_SIZE:
            EVALUATE = True
        else:
            EVALUATE = False
        return EVALUATE