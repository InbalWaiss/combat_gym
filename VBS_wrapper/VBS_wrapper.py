from gym_combat.gym_combat.envs.Common.constants import RED_TYPE, Color, WinEnum, AgentAction

from gym_combat.gym_combat.envs.Greedy import smart_player
from gym_combat.gym_combat.envs.Greedy import Greedy_player

from gym_combat.gym_combat.envs.Arena.Environment import Environment
from gym_combat.gym_combat.envs.Arena.Entity import Entity
from gym_combat.gym_combat.envs.Arena.CState import State
from DsmHandler import DsmHandler, UTMPosition, PixelPosition

from stable_baselines3 import PPO
import os

Env = Environment(False, 'VBS_Wrapper', combat_env_num=1, move_penalty = -0.01)
dsmHandler = DsmHandler()
whos_turn = Color.Blue
#PPO_model = PPO.load(os.path.join(model_path, model_name))

blue_decision_maker = Greedy_player.Greedy_player()
if RED_TYPE == 'Smart':
    red_decision_maker = smart_player.SmartPlayer()
else:  # 'Greedy'
    red_decision_maker = Greedy_player.Greedy_player()
def init():
    Env.blue_player = Entity(blue_decision_maker)
    Env.red_player = Entity(red_decision_maker)

    print("finished init!")


def blue_step(blue_east_utm, blue_north_utm, blue_up_utm, blue_azimute_utm, red_east_utm, red_north_utm, red_up_utm, red_azimute_utm):
    blue_pos_utm = UTMPosition(easting=blue_east_utm, northing=blue_north_utm)
    blue_position_pixel = dsmHandler.utm_to_pixel(blue_pos_utm)
    Env.blue_player.w = blue_position_pixel.east
    Env.blue_player.h = blue_position_pixel.north

    red_pos_utm = UTMPosition(easting=red_east_utm, northing=red_north_utm)
    red_position_pixel = dsmHandler.utm_to_pixel(red_pos_utm)
    Env.red_player.w = red_position_pixel.east
    Env.red_player.h = red_position_pixel.north

    observation_for_blue: State = Env.get_observation_for_blue()
    action_blue: AgentAction = blue_decision_maker.get_action(observation_for_blue)
    Env.take_action(Color.Blue, action_blue)
    #blue_new_position = model.predict(observation_blue)

    new_blue_pos_pixel = PixelPosition(north= Env.blue_player.h, east=Env.blue_player.w)
    new_blue_pos_utm = dsmHandler.pixel_to_utm(new_blue_pos_pixel)

    whos_turn = Color.Blue

    return new_blue_pos_utm.easting, new_blue_pos_utm.northing, -1, 0

def red_step(blue_east_utm, blue_north_utm, blue_up_utm, blue_azimute_utm, red_east_utm, red_north_utm, red_up_utm, red_azimute_utm):
    blue_pos_utm = UTMPosition(easting=blue_east_utm, northing=blue_north_utm)
    blue_position_pixel = dsmHandler.utm_to_pixel(blue_pos_utm)
    Env.blue_player.w = blue_position_pixel.east
    Env.blue_player.h = blue_position_pixel.north

    red_pos_utm = UTMPosition(easting=red_east_utm, northing=red_north_utm)
    red_position_pixel = dsmHandler.utm_to_pixel(red_pos_utm)
    Env.red_player.w = red_position_pixel.east
    Env.red_player.h = red_position_pixel.north

    observation_for_red: State = Env.get_observation_for_red()
    action_red: AgentAction = red_decision_maker.get_action(observation_for_red)
    Env.take_action(Color.Red, action_red)

    new_red_pos_pixel = PixelPosition(north= Env.red_player.h, east=Env.red_player.w)
    new_red_pos_utm = dsmHandler.pixel_to_utm(new_red_pos_pixel)

    whos_turn = Color.Red

    return new_red_pos_utm.easting, new_red_pos_utm.northing, -1, 0

def is_terminal(blue_x_utm, blue_y_utm, blue_z_utm, blue_azimute_utm, red_x_utm, red_y_utm, red_z_utm, red_azimute_utm):
    blue_position_utm = UTMPosition(easting=blue_x_utm, northing=blue_y_utm)
    blue_position_pixel = dsmHandler.utm_to_pixel(blue_position_utm)

    red_position_utm = UTMPosition(easting=red_x_utm, northing=red_y_utm)
    red_position_pixel = dsmHandler.utm_to_pixel(red_position_utm)
    Env.blue_player.w = blue_position_pixel.east
    Env.blue_player.h = blue_position_pixel.north
    Env.red_player.w = red_position_pixel.east
    Env.red_player.h = red_position_pixel.north
    Env.compute_terminal(whos_turn=whos_turn)
    if Env.win_status == WinEnum.NoWin:
        return False
    return True

def choose_start_points():
    Env.reset_players_positions(episode_number=0)
    blue_pos_pixel = PixelPosition(north=Env.blue_player.h, east=Env.blue_player.w)
    blue_pos_utm = dsmHandler.pixel_to_utm(blue_pos_pixel)

    red_pos_pixel = PixelPosition(north =  Env.red_player.h, east = Env.red_player.w)
    red_pos_utm = dsmHandler.pixel_to_utm(red_pos_pixel)

    return blue_pos_utm.easting, blue_pos_utm.northing, -1, 0, red_pos_utm.easting, red_pos_utm.northing, -1, 0,


# class VBS_Wrapper():
#     def __init__(self):
#         #self.env = Environment(False, 'VBS_Wrapper', combat_env_num=1, move_penalty = -0.01)
#         self.env = Env
#         self.dsmHandler = DsmHandler()
#
#         if RED_TYPE == 'Smart':
#             self.red_decision_maker = smart_player.SmartPlayer()
#         else: # 'Greedy'
#             self.red_decision_maker = Greedy_player.Greedy_player()
#
#         self.env.red_player = Entity(self.red_decision_maker)
#
#         self.episode_number = 0
#         self.train_mode = False
#
#
#     def blue_step(self, blue_x_utm, blue_y_utm, blue_z_utm, blue_azimute_utm, red_x_utm, red_y_utm, red_z_utm, red_azimute_utm):
#         blue_position_pixel = self.utm_to_pixel(blue_position_utm)
#         red_position_pixel = self.utm_to_pixel(red_position_utm)
#         blue_new_position = self.env.blue_player.  .get_action(blue_position_pixel, red_position_pixel)
#         #blue_new_position = model.predict(observation_blue)
#         WP = get_new_position_in_utm(new_position)
#         return WP
#
#     def red_step(self, blue_x_utm, blue_y_utm, blue_z_utm, blue_azimute_utm, red_x_utm, red_y_utm, red_z_utm, red_azimute_utm):
#         blue_position_pixel = self.utm_to_pixel(blue_position_geo)
#         red_position_pixel = self.utm_to_pixel(red_position_gro)
#         red_new_position = self.red_decision_maker.get_action(blue_position_pixel, red_position_pixel)
#
#     def step(self, blue_position_geo, red_position_gro):
#         blue_position_pixel = self.geo_to_pixel(blue_position_geo)
#         red_position_pixel = self.geo_to_pixel(red_position_gro)
#         new_position = blue_position.get_action(blue_position_pixel, red_position_pixel)
#         WP = get_new_position_in_geo(new_position)
#         return [WP]
#
#
#     def utm_to_pixel(self, geo_pos):
#         pixel_pos = self.dsmHandler.UTM_to_Pixel(geo_pos)
#
#         return pixel_pos

if __name__ == '__main__':
    init()
    print("new start positions: ", choose_start_points())
    print("blue step: ", blue_step(694294.00, 3588100.00, -1, 0, 694295.00, 3588200.00, -1, 0))
    print("red step: ", red_step(694294.00, 3588100.00, -1, 0, 694295.00, 3588200.00, -1, 0))

    print("is_terminal False: ", is_terminal(694293.00, 3588050.00, -1, 0, 694393.00, 3587950.00, -1, 0))
    print("is_terminal True: ", is_terminal(694294.00, 3588100.00, -1, 0, 694295.00, 3588200.00, -1, 0))

    print()