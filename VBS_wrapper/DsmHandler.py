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
        N = int(np.round(self._top_left_corner_utm[1] - utm_pos.northing))
        if N<0:
            N = np.max([0, N])
        if N>99:
            N = np.min([99, N])

        W = int(np.round(utm_pos.easting - self._top_left_corner_utm[0]))
        if W<0:
            W = np.max([0, W])
        if W>99:
            W = np.min([99, W])

        pixel_pos = PixelPosition(north=N, east=W)

        return pixel_pos


    def pixel_to_utm(self, pixel_pos: PixelPosition):
        add_one_north = 000010.00
        add_one_east = -000010.00


        utm_east = self._top_left_corner_utm[0]+pixel_pos.east*add_one_east
        utm_north = self._top_left_corner_utm[1]+pixel_pos.north*add_one_north

        utm_pos = UTMPosition(easting=utm_east, northing=utm_north)

        return utm_pos



# import matplotlib.pyplot as plt
#
# plt.matshow(DSM)
# plt.show()




