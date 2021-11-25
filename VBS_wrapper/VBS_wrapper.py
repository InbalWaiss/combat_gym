from gym_combat.gym_combat.envs.Common.constants import RED_TYPE, Color, WinEnum, AgentAction, DSM

from gym_combat.gym_combat.envs.Greedy import smart_player
from gym_combat.gym_combat.envs.Greedy import Greedy_player

from gym_combat.gym_combat.envs.Arena.Environment import Environment
from gym_combat.gym_combat.envs.Arena.Entity import Entity
from gym_combat.gym_combat.envs.Arena.CState import State
from DsmHandler import DsmHandler, UTMPosition, PixelPosition
import matplotlib.pyplot as plt


from stable_baselines3 import PPO
import os

DEBUG = False


class FPSWrapper():
    def init(self):
        self.Env = Environment(False, 'VBS_Wrapper', combat_env_num=1, move_penalty=-0.01)
        self.dsmHandler = DsmHandler()
        self.whos_turn = Color.Blue
        # PPO_model = PPO.load(os.path.join(model_path, model_name))

        self.blue_decision_maker = Greedy_player.Greedy_player()
        if RED_TYPE == 'Smart':
            self.red_decision_maker = smart_player.SmartPlayer()
        else:  # 'Greedy'
            self.red_decision_maker = Greedy_player.Greedy_player()
        self.Env.blue_player = Entity(self.blue_decision_maker)
        self.Env.red_player = Entity(self.red_decision_maker)


    def blue_step(self, blue_east_utm, blue_north_utm, blue_up_utm, blue_azimute_utm, red_east_utm, red_north_utm, red_up_utm, red_azimute_utm):
        blue_pos_utm = UTMPosition(easting=blue_east_utm, northing=blue_north_utm)
        blue_position_pixel = self.dsmHandler.utm_to_pixel(blue_pos_utm)
        self.Env.blue_player.w = blue_position_pixel.east
        self.Env.blue_player.h = blue_position_pixel.north

        red_pos_utm = UTMPosition(easting=red_east_utm, northing=red_north_utm)
        red_position_pixel = self.dsmHandler.utm_to_pixel(red_pos_utm)
        self.Env.red_player.w = red_position_pixel.east
        self.Env.red_player.h = red_position_pixel.north

        if DEBUG:
            import copy
            DSM_copy = copy.deepcopy(DSM)
            DSM_copy[self.Env.blue_player.w, self.Env.blue_player.h] = 10
            DSM_copy[self.Env.red_player.w, self.Env.red_player.h] = 20
            plt.matshow(DSM)
            plt.show()

        observation_for_blue: State = self.Env.get_observation_for_blue()
        action_blue: AgentAction = self.blue_decision_maker.get_action(observation_for_blue)
        self.Env.take_action(Color.Blue, action_blue)
        #blue_new_position = model.predict(observation_blue)

        if DEBUG:
            import copy
            DSM_copy = copy.deepcopy(DSM)
            DSM_copy[self.Env.blue_player.h, self.Env.blue_player.w] = 10
            DSM_copy[self.Env.red_player.h, self.Env.red_player.w] = 20
            plt.matshow(DSM_copy)
            plt.show()

        new_blue_pos_pixel = PixelPosition(north= self.Env.blue_player.h, east=self.Env.blue_player.w)
        new_blue_pos_utm = self.dsmHandler.pixel_to_utm(new_blue_pos_pixel)

        self.whos_turn = Color.Blue

        return new_blue_pos_utm.easting, new_blue_pos_utm.northing, -1, 0

    def red_step(self, blue_east_utm, blue_north_utm, blue_up_utm, blue_azimute_utm, red_east_utm, red_north_utm, red_up_utm, red_azimute_utm):

        blue_pos_utm = UTMPosition(easting=blue_east_utm, northing=blue_north_utm)
        blue_position_pixel = self.dsmHandler.utm_to_pixel(blue_pos_utm)
        self.Env.blue_player.w = blue_position_pixel.east
        self.Env.blue_player.h = blue_position_pixel.north

        red_pos_utm = UTMPosition(easting=red_east_utm, northing=red_north_utm)
        red_position_pixel = self.dsmHandler.utm_to_pixel(red_pos_utm)
        self.Env.red_player.w = red_position_pixel.east
        self.Env.red_player.h = red_position_pixel.north

        if DEBUG:
            import copy
            DSM_copy = copy.deepcopy(DSM)
            DSM_copy[self.Env.blue_player.h, self.Env.blue_player.w] = 10
            DSM_copy[self.Env.red_player.h, self.Env.red_player.w] = 20
            plt.matshow(DSM)
            plt.show()

        observation_for_red: State = self.Env.get_observation_for_red()
        action_red: AgentAction = self.red_decision_maker.get_action(observation_for_red)
        self.Env.take_action(Color.Red, action_red)

        if DEBUG:
            import copy
            DSM_copy = copy.deepcopy(DSM)
            DSM_copy[self.Env.blue_player.h, self.Env.blue_player.w] = 10
            DSM_copy[self.Env.red_player.h, self.Env.red_player.w] = 20
            plt.matshow(DSM)
            plt.show()

        new_red_pos_pixel = PixelPosition(north= self.Env.red_player.h, east=self.Env.red_player.w)
        new_red_pos_utm = self.dsmHandler.pixel_to_utm(new_red_pos_pixel)

        self.whos_turn = Color.Red

        return new_red_pos_utm.easting, new_red_pos_utm.northing, -1, 0

    def is_terminal(self, blue_x_utm, blue_y_utm, blue_z_utm, blue_azimute_utm, red_x_utm, red_y_utm, red_z_utm, red_azimute_utm):

        blue_position_utm = UTMPosition(easting=blue_x_utm, northing=blue_y_utm)
        blue_position_pixel = self.dsmHandler.utm_to_pixel(blue_position_utm)

        red_position_utm = UTMPosition(easting=red_x_utm, northing=red_y_utm)
        red_position_pixel = self.dsmHandler.utm_to_pixel(red_position_utm)
        self.Env.blue_player.w = blue_position_pixel.east
        self.Env.blue_player.h = blue_position_pixel.north
        self.Env.red_player.w = red_position_pixel.east
        self.Env.red_player.h = red_position_pixel.north
        self.Env.compute_terminal(whos_turn=self.whos_turn)

        if DEBUG:
            import copy
            DSM_copy = copy.deepcopy(DSM)
            DSM_copy[self.Env.blue_player.h, self.Env.blue_player.w] = 10
            DSM_copy[self.Env.red_player.h, self.Env.red_player.w] = 20
            plt.matshow(DSM)
            plt.show()

        if self.Env.win_status == WinEnum.NoWin:
            return False
        return True

    def new_start_positions(self):
        self.Env.reset_players_positions(episode_number=0)
        blue_pos_pixel = PixelPosition(north=self.Env.blue_player.h, east=self.Env.blue_player.w)
        blue_pos_utm = self.dsmHandler.pixel_to_utm(blue_pos_pixel)

        red_pos_pixel = PixelPosition(north =  self.Env.red_player.h, east = self.Env.red_player.w)
        red_pos_utm = self.dsmHandler.pixel_to_utm(red_pos_pixel)

        if DEBUG:
            import copy
            DSM_copy = copy.deepcopy(DSM)
            DSM_copy[self.Env.blue_player.h][self.Env.blue_player.w] = 10
            DSM_copy[self.Env.red_player.h][self.Env.red_player.w] = 20
            plt.matshow(DSM_copy)
            plt.show()

        return blue_pos_utm.easting, blue_pos_utm.northing, -1, 0, red_pos_utm.easting, red_pos_utm.northing, -1, 0,



if __name__ == '__main__':
    fpswrapper = FPSWrapper()
    fpswrapper.init()
    print("new start positions: ", fpswrapper.new_start_positions())
    print("blue step: ", fpswrapper.blue_step(694303.00, 3588040.00, -1, 0, 694383.00, 3587960.00, -1, 0))
    print("red step: ", fpswrapper.red_step(694303.00, 3588040.00, -1, 0, 694383.00, 3587960.00, -1, 0))

    print("is_terminal False: ", fpswrapper.is_terminal(694303.00, 3588040.00, -1, 0, 694383.00, 3587960.00, -1, 0))
    print("is_terminal True: ", fpswrapper.is_terminal(694343.00, 3588000.00, -1, 0, 694343.00, 3588010.00, -1, 0))

