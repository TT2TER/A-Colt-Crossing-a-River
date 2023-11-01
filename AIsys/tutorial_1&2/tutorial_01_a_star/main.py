import json
from swarmae_api import SwarmaeApiClass
import Astar
import cv2
import numpy as np


def main():

    swarmeapi = SwarmaeApiClass()
 
    #获取路网信息 PIL格式
    img, _, world_offset, _, _ = swarmeapi.game.get_road_network()
    maps = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    maps = cv2.cvtColor(maps,cv2.COLOR_BGR2GRAY)
    #获取车辆节点
    sw_node = swarmeapi.sw_node[1]
    x,y,_,_=sw_node.get_location()
    x=int(x-world_offset[0])
    y=int(y-world_offset[1])
    end_x=1800
    end_y=2010
    path=Astar.Astar(maps,x,y,end_x,end_y)
    #print(path)

if __name__ == '__main__':
    main()

