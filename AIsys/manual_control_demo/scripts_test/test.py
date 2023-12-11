from swarmae.SwarmAEClient import SwarmAEClient
from lib.vehicle import Vehicle

# 建立链接
AeClient = SwarmAEClient(ue_ip="localhost", ue_port=2000)

#创建结点，控制车辆
Vehicle_1 = Vehicle(Client=AeClient, vehicle_name="UGV_1", vehicle_no=1)

run = True
while run:
    Vehicle_1.longitudinal_control(700000, 0)