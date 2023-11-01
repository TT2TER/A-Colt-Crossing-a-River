import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def Astar(maps,start_x,start_y,end_x,end_y):
    print('开始规划')

    maps_size=np.array(maps)#获取图像行和列大小
    hight=maps_size.shape[0]#行数->y
    width=maps_size.shape[1]#列数->x

    start={'位置':(start_x,start_y),'G':0,'F':0,'父节点':(start_x,start_y)}#起点
    end={'位置':(end_x,end_y),'G':700,'F':700,'父节点':(end_x,end_y)}#终点
    print(start_x,start_y)

    openlist=[]#open列表，存储可能路径
    closelist=[start]#close列表，已走过路径
    step_size=5#搜索步长。

    #步长太小，搜索速度就太慢。步长太大，可能直接跳过障碍，得到错误的路径
    #步长大小要大于图像中最小障碍物宽度
    while 1:
        s_point=closelist[-1]['位置']#获取close列表最后一个点位置，S点 (x,y)
        G=closelist[-1]['G']
        add=([0,step_size],[0,-step_size],[step_size,0],[-step_size,0])#可能运动的四个方向增量
        for i in range(len(add)):
            x=s_point[0]+add[i][0]#检索超出图像大小范围则跳过
            if x<0 or x>=width:
                continue
            y=s_point[1]+add[i][1]
            if y<0 or y>=hight:#检索超出图像大小范围则跳过
                continue
            if maps[y][x]==0:
                continue
            G=G+1#计算代价
            H=abs(x-end['位置'][0])+abs(y-end['位置'][1])#计算代价
            F=G+H

            if H<20:#当逐渐靠近终点时，搜索的步长变小
                step_size=1
            addpoint={'位置':(x,y),'G':G,'F':F,'父节点' :s_point}#更新位置
            count=0
            for i in openlist:
                if i['位置']==addpoint['位置']:
                    count+=1
                    if i['F']>addpoint['F']:
                        i['G']=addpoint['G']
                        i['F']=addpoint['F']
                        i['父节点']=s_point
            for i in closelist:
                if i['位置']==addpoint['位置']:
                    count+=1
            if count==0:#新增点不在open和close列表中
                 openlist.append(addpoint)

        t_point={'位置':(91,70),'G':10000,'F':10000,'父节点':(91,70)}
        for j in range(len(openlist)):#寻找代价最小点
            if openlist[j]['F']<t_point['F']:
                t_point=openlist[j]
                #print(1)
        for j in range(len(openlist)):#在open列表中删除t点
            if t_point==openlist[j]:
                openlist.pop(j)
                break
        closelist.append(t_point)#在close列表中加入t点
        
        if t_point['位置']==end['位置']:#找到终点！！
            print("找到终点")
            break
 
    
    #逆向搜索找到路径
    road=[]
    road.append(closelist[-1])
    point=road[-1]
    
    while 1:
        for i in closelist:
            if i['位置']==point['父节点']:#找到父节点
                point=i
                #print(point)
                road.append(point)
        if point==start:
            print("路径搜索完成")
            break
    
    informap = cv2.cvtColor(maps, cv2.COLOR_GRAY2BGR)
    path = [item['位置'] for item in road]
 
    for i in range(len(path)):#整数型画路径
         cv2.circle(informap,path[i],2,(0,0,200),-1)    
    cv2.circle(informap,start['位置'],15,(0,255,0),-1)#起点
    cv2.circle(informap,end['位置'],15,(0,0,255),-1)#终点

    target_width = 900
    target_height = 900
    cv2.imwrite("informap.png",informap)
    informap = cv2.resize(informap, (target_width, target_height))
    
    cv2.imshow("informap.png",informap)
    cv2.waitKey(0)

    return path


        
 