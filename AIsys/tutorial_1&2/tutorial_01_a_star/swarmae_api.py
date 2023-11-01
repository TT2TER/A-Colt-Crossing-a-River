from swarmae.SwarmAEClient import SwarmAEClient
import json
import time


# 负责节点的注册
class SwarmaeApiClass(object):
    sw_code = {}
    sw_node = {}

    def __init__(self, ue_ip="127.0.0.1", ue_port=2000):
        self.ae_client = SwarmAEClient(ue_ip=ue_ip, ue_port=ue_port)
        timestamp, _, code = self.ae_client.get_world()
        timestamp, game, code = self.ae_client.get_game()
        self.game = game
        self.task_areas = None
        self.create_vehicles()


    # node类
    def creat_node(self, ae_client, id, node_type):
        frame_timestamp, sw_node, sw_code = ae_client.register_node(
            node_type=node_type,
            node_name="节点" + str(id),
            node_no=id,
            frame_timestamp=int(round(time.time() * 1000))
        )
        if sw_code == 200:
            node_name, node_no, team, node_type, _, _ = sw_node.get_node_info()
            print("--Clinet [", id, "]: ", node_name, node_no, team, node_type)
        else:
            print("--Clinet [", id, "]: ", frame_timestamp, sw_code)
        return sw_code, sw_node

    # 注册载具节点
    def create_vehicles(self):
        code, node = self.creat_node(self.ae_client, 1, '四轮车')
        if node is not None:
            self.sw_code[1] = code
            self.sw_node[1] = node

