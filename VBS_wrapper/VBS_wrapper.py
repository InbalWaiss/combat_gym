
import DsmHandler

class VBS_Wrapper():
    def __init__(self, path_to_dsm):
        self.dsmHandler = DsmHandler(path_to_dsm)
        # load DSM
        # create env and load network
        pass


    def step(self, blue_position_geo, red_position_gro):
        blue_position_pixel = self.geo_to_pixel(blue_position_geo)
        red_position_pixel = self.geo_to_pixel(red_position_gro)
        new_position = blue_position.get_action(blue_position_pixel, red_position_pixel)
        WP = get_new_position_in_geo(new_position)
        return [WP]


    def geo_to_pixel(self, geo_pos):
        pixel_pos = self.dsmHandler.GEO_to_Pixel(geo_pos)

        return pixel_pos



if __name__ == '__main__':
    path = None #TODO
    vbs_fps = VBS_Wrapper(path)
