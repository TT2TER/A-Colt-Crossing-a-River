import time
import cv2
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
import lib.mine_sweep_planner as msp
import lib.swarm_api as api

def plot_helper(vehicle_pose, ref_path, vehicle_geometry, actual_x_list, actual_y_list):
    area = 15
    vehicle_shape = vehicle_geometry.get_vehicle_shape(vehicle_pose)

    plt.cla()
    plt.plot(ref_path.interp_x_, ref_path.interp_y_, color='black', label='reference path')
    plt.plot(vehicle_shape[:,0], vehicle_shape[:,1], color='orange', label='ego_vehicle')
    plt.plot(actual_x_list, actual_y_list, color='red', label='traveld path')

    plt.axis('equal')
    plt.xlim([vehicle_pose.x_ - area, vehicle_pose.x_ + area])
    plt.ylim([vehicle_pose.y_ - area, vehicle_pose.y_ + area])
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)

if __name__ == '__main__':
    parameters = param_parser.Parameters(".\config\params.ini")

    AeClient = SwarmAEClient(ue_ip="localhost", ue_port=2000)
    _, AeGame, _ = AeClient.get_game()
    AeGame.stage_start('reconnasissance_start')
    AeGame.stage_complete('reconnaissance_end')
    AeGame.stage_start('vau_reconnasissance_start')
    AeGame.stage_complete('vau_reconnaissance_end')
    map_image, scale, world_offset, _, _ = AeGame.get_road_network()
    map_image = np.array(map_image.convert('L'))
    tmp_image = map_image.copy()

    #将读到的地图进行缩放，缩放比例为scale
    # print("scale: ", scale)
    # print("world_offset: ", world_offset)
    offset_x = world_offset[0] / scale
    offset_y = world_offset[1] / scale

    expansion_radius = 3

    #TODO：这个-640等是怎么来的
    for i in range(int(-640 - offset_y), int(-140 - offset_y)):
        for j in range(int(-780 - offset_x), int(50 - offset_x)):
            # 将所有目前距离0像素点膨胀半径以内的点都设置为0
            if map_image[i][j] == 0:
                for k in range(i - expansion_radius, i + expansion_radius + 1):
                    for l in range(j - expansion_radius, j + expansion_radius + 1):
                        if k >= 0 and k < map_image.shape[0] and l >= 0 and l < map_image.shape[1]:
                            tmp_image[k][l] = 0
    map_image = tmp_image.copy()

    for i in range(int(-640 - offset_y), int(-614 - offset_y)):
        map_image[i][int(-780 - offset_x)] = 0
    for i in range(int(-640 - offset_y), int(-629 - offset_y)):
        for j in range(int(-780 - offset_x), int(-750 - offset_x)):
            map_image[i][j] = 0

    #将map_image输出为png文件
    # cv2.imwrite('map_image.png', map_image)
    # exit()


    # map_mine = map_image[int(-614 + offset_y) : int(-155.7 + offset_y), int(-758 + offset_x) : int(27.7 + offset_x)]
    # # 将map_mine 输出成csv文件
    # np.savetxt('map_mine.csv', map_mine, delimiter = ',')
    # exit()

    vehicle = api.Vehicle(AeClient,
                        vehicle_name="无人车",
                        vehicle_no=4)
    
    Mplan = msp.MineSweepPlanner(vehicle, scale, world_offset, map_image)
    vehicle_pose = vehicle.get_transform()
    point_list = [(vehicle_pose.x_, vehicle_pose.y_)]
    point_list.append((34.7, vehicle_pose.y_))
    point_list.append((35.7, -360))
    point_list.append((-760, -360))
    point_list.append((-760, -463))
    point_list.append((34.7, -463))
    point_list.append((35.7, -551))
    point_list.append((-760, -553))
    point_list.append((-760, -620))
    point_list.append((-540, -620))
    point_list.append((-540, -148))
    point_list.append((-197, -148))
    point_list.append((-197, -640))
    for point in point_list:
        if map_image[int(point[1] - offset_y)][int(point[0] - offset_x)] == 0:
            print(point)
            exit()
    reference_x, reference_y = Mplan.plan(point_list)
    ref_path = fop.ReferencePath(reference_x, reference_y, 0.1)
    # for i in range(ref_path.interp_x_.shape[0]):
    #     print(i,(ref_path.interp_x_[i, 0], ref_path.interp_y_[i, 0]))
    goal_x = ref_path.interp_x_[-1, 0]
    goal_y = ref_path.interp_y_[-1, 0]

    stanley = controller.Stanley(parameters.K_)
    pid = controller.PID(parameters.Kp_, 
                         parameters.Ki_, 
                         parameters.Kd_)
    
    vehicle_length = 3
    vehicle_width = 2
    wheel_base = 0.85 * vehicle_length
    vehicle_geometry = fop.VehicleGeometry(l=vehicle_length,w=vehicle_width)

    initial_x = ref_path.interp_x_[0,0]
    initial_y = ref_path.interp_y_[0,0]
    initial_theta = ref_path.interp_theta_[0,0]

    initial_kappa=0
    initial_v=0
    initial_a=0

    planner = fop.FrenetOptimalPlanner(ref_path,parameters)
    vehicle_pose = struct.Transform(initial_x,initial_y,initial_theta)
    vehicle_state = struct.State(vehicle_pose,initial_v,initial_a,initial_kappa)
    frenet_state = fop.transform_cartesian_to_frenet(vehicle_state,ref_path)
    optimal_trajectory, valid_trajectory, trajectory_list = planner.update_planning(frenet_state, vehicle_geometry, [])

    actual_x_list = []
    actual_y_list = []
    actual_v_list = []
    actual_theta_list = []

    prev_time = time.time()
    dt = 0.1


    while True:
        # 如果到达了终点，就将车辆刹停，并跳出循环
        if optimal_trajectory.s_[0] >= ref_path.interp_s_[-1]:
            actual_x_list.append(vehicle_state.pose_.x_)
            actual_y_list.append(vehicle_state.pose_.y_)
            actual_v_list.append(vehicle_state.v_)
            vehicle.apply_control(steer=0,throttle=0,brake=0.8, hand_brake=False)
            time.sleep(10)
            # 然后拉手刹制动
            vehicle.apply_control(steer=0, throttle=0, brake=0, hand_brake=True)
            print("\nFinish")
            break
        
        # 记录数据
        actual_x_list.append(vehicle_state.pose_.x_)
        actual_y_list.append(vehicle_state.pose_.y_)

        # 实时绘图
        plot_helper(vehicle_pose, ref_path, vehicle_geometry, actual_x_list, actual_y_list)

        # 寻找控制器参考点（一些工程经验，虽然 Stanley 的原理上是要找前轴投影点，但往往找一个更远的投影点可以在
        # 车辆高速行驶时获得更好的效果，此外，stanley 控制中的 delta_y 和 delta_theta 的参与比例都可以调整，为
        # 了调参方便，大家可以参考 Tutorial04 和 Tutorial05 将所有参数封装到 ini 文件中进行处理）
        front_x = vehicle_state.pose_.x_ + wheel_base * math.cos(vehicle_state.pose_.theta_)
        front_y = vehicle_state.pose_.y_ + wheel_base * math.sin(vehicle_state.pose_.theta_)

        dx = np.ravel(optimal_trajectory.x_) - front_x
        dy = np.ravel(optimal_trajectory.y_) - front_y
        dist = dx ** 2 + dy ** 2

        min_dist_idx = np.argmin(dist)
        min_dist = np.min(dist)
        
        target_x = optimal_trajectory.x_[min_dist_idx,0]
        target_y = optimal_trajectory.y_[min_dist_idx,0]
        target_theta = optimal_trajectory.theta_[min_dist_idx,0]
        target_v = optimal_trajectory.v_[min_dist_idx,0]
        target_pose = struct.Transform(target_x,target_y,target_theta)

        # 计算控制量
        delta = stanley.update_control(target_pose, vehicle_state)
        accel = pid.update_control(target_v, vehicle_state.v_, dt)

        # 将转向归一化到 [-pi/6, pi/6]
        steer = delta / (math.pi / 3)

        # 将加速度转换为油门、刹车
        if accel > 0:
            throttle = accel
            brake = 0
        else:
            throttle = 0
            brake = -accel * 1.5
        
        # 控制车辆运动
        vehicle.apply_control(steer, throttle, brake, hand_brake=False)

        # 更新采样时间
        dt = time.time() - prev_time
        prev_time = time.time()
        
        # 更新车辆状态信息
        vehicle_pose = vehicle.get_transform()
        vehicle_pose = struct.Transform(vehicle_pose.x_, vehicle_pose.y_, vehicle_pose.theta_ / 180 * math.pi)
        vehicle_imu  = vehicle.get_imu_data()
        kappa = math.tan(delta) / wheel_base
        vehicle_state = struct.State(vehicle_pose, abs(vehicle_imu.v_), vehicle_imu.a_, kappa)

        dx = np.ravel(optimal_trajectory.x_) - vehicle_state.pose_.x_
        dy = np.ravel(optimal_trajectory.y_) - vehicle_state.pose_.y_
        dist = dx ** 2 + dy ** 2
        min_dist_idx = np.argmin(dist)
        min_dist = np.min(dist)

        frenet_state=struct.FrenetState(
            optimal_trajectory.s_[min_dist_idx,0],
            optimal_trajectory.s_dot_[min_dist_idx,0],
            optimal_trajectory.s_ddot_[min_dist_idx,0],
            optimal_trajectory.d_[min_dist_idx,0],
            optimal_trajectory.d_dot_[min_dist_idx,0],
            optimal_trajectory.d_ddot_[min_dist_idx,0]
            )
        optimal_trajectory, valid_trajectory, trajectory_list = planner.update_planning(frenet_state, vehicle_geometry, [])