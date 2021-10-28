import pymap3d


class DsmHandelr():
    def __init__(self, path_to_DSM=None):
        x=1
        # load DSM
        pass

    def GEO_to_Pixel(self, geo_pos):
        lat = geo_pos[0]
        lon = geo_pos[1]
        pymap3d.geodetic2enu(lat, lon, h, lat0, lon0, h0)



if __name__ == '__main__':


    # The local coordinate origin (Zermatt, Switzerland)
    lat0 = 46.017 # deg
    lon0 = 7.750  # deg
    h0 = 1673     # meters

    # The point of interest
    lat = 45.976  # deg
    lon = 7.658   # deg
    h = 4531      # meters

    print(pymap3d.geodetic2enu(lat, lon, h, lat0, lon0, h0))



    # create an ellipsoid object
    ell_clrk66 = pymap3d.Ellipsoid()#'clrk66')
    # print ellipsoid's properties
    print(ell_clrk66)

    # output
    #(6378206.4, 6356583.8, 0.0033900753039287634)

    lat0, lon0, h0 = 5.0, 48.0, 10.0   # origin of ENU, (h is height above ellipsoid)
    e1, n1, u1     =  0.0,  0.0,  0.0  # ENU coordinates of test point, `point_1`
    # From ENU to geodetic computation
    # lat1, lon1, h1 = pymap3d.enu2geodetic(e1, n1, u1, \
    #                                       lat0, lon0, h0, \
    #                                       ell=ell_clrk66, deg=True)  # use clark66 ellisoid
    lat1, lon1, h1 = pymap3d.enu2geodetic(e1, n1, u1, lat0, lon0, h0)

    print(lat1, lon1, h1)


