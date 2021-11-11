import pymap3d
import numpy as np
from combat_gym.gym_combat.gym_combat.envs.Common.constants import DSM


class UTMPosition():
    easting : float #[m]
    northing : float #[m]

    def to_numpy_array(self) -> np.ndarray:
        return np.ndarray([self.easting, self.northing])

class PixelPosition():
    north : int #[m]
    east : int #[m]

    def to_numpy_array(self) -> np.ndarray:
        return np.ndarray([self.east, self.north])


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





        # load DSM
        pass

    def GEO_to_Pixel(self, geo_pos):
        lat = geo_pos[0]
        lon = geo_pos[1]
        pymap3d.geodetic2enu(lat, lon, h, lat0, lon0, h0)


    def utm_to_pixel(self, utm_pos: UTMPosition):
        N = UTMPosition.northing - self._top_left_corner_utm[0]
        W = self._top_left_corner_utm[1]-utm_pos.easting

    def pixel_to_utm(self, pixel_pos: PixelPosition):
        add_one_north = 000010.00
        add_one_east = -000010.00

import matplotlib.pyplot as plt

plt.matshow(DSM)
plt.show()


#if __name__ == '__main__':




    # # The local coordinate origin (Zermatt, Switzerland)
    # lat0 = 46.017 # deg
    # lon0 = 7.750  # deg
    # h0 = 1673     # meters
    #
    # # The point of interest
    # lat = 45.976  # deg
    # lon = 7.658   # deg
    # h = 4531      # meters
    #
    # print(pymap3d.geodetic2enu(lat, lon, h, lat0, lon0, h0))
    #
    #
    #
    # # create an ellipsoid object
    # ell_clrk66 = pymap3d.Ellipsoid()#'clrk66')
    # # print ellipsoid's properties
    # print(ell_clrk66)
    #
    # # output
    # #(6378206.4, 6356583.8, 0.0033900753039287634)
    #
    # lat0, lon0, h0 = 5.0, 48.0, 10.0   # origin of ENU, (h is height above ellipsoid)
    # e1, n1, u1     =  0.0,  0.0,  0.0  # ENU coordinates of test point, `point_1`
    # # From ENU to geodetic computation
    # # lat1, lon1, h1 = pymap3d.enu2geodetic(e1, n1, u1, \
    # #                                       lat0, lon0, h0, \
    # #                                       ell=ell_clrk66, deg=True)  # use clark66 ellisoid
    # lat1, lon1, h1 = pymap3d.enu2geodetic(e1, n1, u1, lat0, lon0, h0)
    #
    # print(lat1, lon1, h1)


