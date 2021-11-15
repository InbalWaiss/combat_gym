import pymap3d
import numpy as np
from combat_gym.gym_combat.gym_combat.envs.Common.constants import DSM


class UTMPosition():
    def __init__(self, northing, easting):
        self.northing = northing
        self.easting = easting

    def to_numpy_array(self) -> np.ndarray:
        return np.ndarray([self.easting, self.northing])

class PixelPosition():
    def __init__(self, north, east):
        self.north = north
        self.east = east



    def to_numpy_array(self) -> np.ndarray:
        return np.ndarray([self.north, self.east])


class DsmHandler():
    def __init__(self, path_to_DSM=None):
        self._top_left_corner_utm = (694293.00, 3588050.00)
        self._top_right_corner_utm = (694393.00, 3588050.00)
        self._bottom_left_corner_utm = (694293.00, 3587950.00)
        self._bottom_right_corner_utm = (694393.00, 3587950.00)

        self._top_left_corner_pixel = (0, 0)
        self._top_right_corner_pixel = (0, 99)
        self._bottom_left_corner_pixel = (99, 0)
        self._bottom_right_corner_pixel = (99, 99)



    def utm_to_pixel(self, utm_pos: UTMPosition):
        N = utm_pos.northing - self._top_left_corner_utm[0]
        W = self._top_left_corner_utm[1]-utm_pos.easting

        pixel_pos = PixelPosition(north=N, east=W)

        return pixel_pos


    def pixel_to_utm(self, pixel_pos: PixelPosition):
        add_one_north = 000010.00
        add_one_east = -000010.00

        utm_north = self._top_left_corner_utm[0]+pixel_pos.north*add_one_north
        utm_east = self._top_left_corner_utm[0]+pixel_pos.north*add_one_east

        utm_pos = UTMPosition(northing=utm_north, easting=utm_east)

        return utm_pos



# import matplotlib.pyplot as plt
#
# plt.matshow(DSM)
# plt.show()




