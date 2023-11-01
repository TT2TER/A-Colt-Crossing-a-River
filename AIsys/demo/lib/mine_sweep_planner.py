import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from swarmae.SwarmAEClient import SwarmAEClient

import lib.frenet_optimal_planner as fop
import lib.utils as utils
import lib.data_struct as struct
import lib.vehicle_simulator as sim
import lib.controller as controller
import lib.param_parser as param_parser
import lib.a_star_planner as asp
import lib.swarm_api as api


class MineSweepPlanner:
    def __init__(self, vehicle, scale, world_offset, img):
        self.vehicle_pose = vehicle.get_transform()
        self.start_x = self.vehicle_pose.x_
        self.start_y = self.vehicle_pose.y_
        self.offset_x = world_offset[0] / scale
        self.offset_y = world_offset[1] / scale
        self.img = img

    def plan(self, point_list=[]):
        reference_x = np.array([self.start_x])
        reference_y = np.array([self.start_y])
        for i in range(1, len(point_list)):
            Aplanner = asp.AStar(int(point_list[i][0] - self.offset_x), int(point_list[i][1] - self.offset_y),
                                 int(point_list[i - 1][0] - self.offset_x), int(point_list[i - 1][1] - self.offset_y), self.img)
            reference_trajectory = Aplanner.update_planning()
            reference_trajectory = np.array(reference_trajectory)
            tmp_x = reference_trajectory[:, 0] + self.offset_x
            tmp_y = reference_trajectory[:, 1] + self.offset_y
            reference_x = np.concatenate((reference_x, tmp_x[1:]))
            reference_y = np.concatenate((reference_y, tmp_y[1:]))
        reference_x = reference_x.reshape(-1, 1)
        reference_y = reference_y.reshape(-1, 1)
        print(reference_x.shape, reference_y.shape)
        return reference_x, reference_y
