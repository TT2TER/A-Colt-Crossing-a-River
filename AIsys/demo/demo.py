## 仿真器启动顺序
# 1. 通过快捷方式启动软件（参考第一次课的教程）
# 2. 启动配置文件 subject3.exe （subject4.exe 的配置文件有点问题，先用 subject3.exe）

import time

# 仿真软件官方的工具包
from swarmae.SwarmAEClient import SwarmAEClient

# 自己封装的函数和接口
import lib.swarm_api as api
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ###########################################################################
    # 仿真器接口测试 
    ###########################################################################

    # 和仿真器建立连接
    AeClient = SwarmAEClient(ue_ip="localhost", ue_port=2000)

    #读取地图比例及偏差
    _, AeGame, _ = AeClient.get_game()
    img, scale, world_offset, _, _ = AeGame.get_road_network()        

    plt.figure()
    plt.imshow(img)
    plt.show()

    print("scale and offset: ", [scale, world_offset])

    # 测试四轮车，这里的 vehicle_no 和仿真器左上角的车辆编号对应
    vehicle = api.Vehicle(AeClient,
                        vehicle_name="李四",
                        vehicle_no=1)

    # 获取车辆位置
    vehicle_pose = vehicle.get_transform()
    print("vehicle_pose: ", [vehicle_pose.x_, vehicle_pose.y_, vehicle_pose.theta_])

    # 获取车辆速度信息
    vehicle_imu = vehicle.get_imu_data()
    print("vehilce_imu: ", [vehicle_imu.a, vehicle_imu.v, vehicle_imu.rate_yaw])

    # 控制车辆移动 10 秒然后拉手刹
    vehicle.apply_control(0.8, 0.5, 0.0, False)
    time.sleep(10)
    vehicle.apply_control(0.0, 0.0, 0.0, True)

    ###########################################################################
    # TODO 算法流程
    #
    # 1. 根据获取的路网图规划出一条全局路径
    # 2. 参考 Lattice Planner Tutorial 部分进行初始化：
    #    2.1 创建用于补全全局路径的 ReferencePath
    #    2.2 创建用于碰撞检测的 VehicleGeometry
    #    2.3 创建 PID 和 Stanley 控制器
    #    2.4 读取车辆当前位置，通过 transform_cartesian_to_frenet 初始化
    #        车辆当前的 frenet 坐标
    #    2.5 创建局部规划器 FrenetOptimalPlanner，并从当前坐标规划一帧局部轨迹
    # 3. 进入循环，开始重复 局部规划-运动控制 的流程
    # 
    # while (车辆当前位置和全局路径终点之间的距离大于阈值) {
    #       寻找车辆前轴到局部路径的参考点
    #       将参考点输入给 Stanley 控制器，计算出转向角 steer
    #       由于 Stanley 计算结果是 -pi/6 到 pi/6 的角度，控制车辆转向时需要归一化到 [-1, 1] 
    #       
    #       将参考速度输入给 PID 控制器，计算出 accel
    #       当 accel 为正时，throttle = accel，brake = 0
    #       当 accel 为负时，throttle = 0    , brake = -accel
    # 
    #       调用 Vehicle 的 apply_control 接口，输入油门、转向等信息，控制车辆移动 
    #       
    #       获取车辆当前位置
    #       从上一帧 FrenetOptimalPlanner 的规划结果中寻找车辆当前位置的投影点
    #       设置上一帧规划结果的投影点为规划起点
    #       调用 FrenetOptimalPlanner.update_planning，规划下一帧轨迹     
    # } 
    #      
    # 流程和 Tutorial-04 / Tutorial-05 完全一致，把这两个教程看明白了，上手准没问题
    # 障碍物探测和武器调用接口我没有封装到 swarm_api.py 中，需要大家自己参考 Vehicle 类
    # 去自行编写代码