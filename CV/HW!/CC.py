import numpy as np
from time import sleep
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#这里是p3d文件路径
path='CV/house.p3d'
# 定义相机内参
setK =[[5, 0, 3], [0, 4, 2], [0, 0, 1]]
# 定义相机外参数
sett=[[0], [0], [0]]
# 定义旋转角度
theta_x = np.radians(45)
theta_y = np.radians(30)
theta_z = np.radians(30)

# 读取p3d文件
def read_p3d(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = [line.split() for line in lines]
    lines = [[float(x) for x in line] for line in lines]
    # 将每一行坐标变为数组形式
    lines = np.array(lines)
    return lines

# 设定相机的内外参数
def set_camera(K, R, t):
    K = np.array(K)
    R = np.array(R)
    t = np.array(t)
    # 将R，t合并为一个矩阵，形式为[R∣t]
    Rt = np.concatenate((R, t), axis=1)
    return K, Rt

# 计算像素坐标
def compute_pixel(K, Rt, lines):
    # 将世界坐标系下的坐标转换为相机坐标系下的坐标
    camera_coor = np.dot(Rt, lines.T)
    # 将相机坐标系下的坐标转换为像素坐标系下的坐标
    pixel_coor = np.dot(K, camera_coor)
    # 将齐次坐标系下的坐标转换为非齐次坐标系下的坐标
    pixel_coor = pixel_coor / pixel_coor[2]
    return pixel_coor.T

if __name__ == '__main__':
    lines = read_p3d(path)
    # 将读取到的坐标变为齐次坐标系形式
    lines = np.concatenate((lines, np.ones((len(lines), 1))), axis=1)

    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]])
    # 计算总旋转矩阵 Ro
    Ro = np.dot(np.dot(Rx, Ry), Rz)
    # 设定相机的内外参数
    K, Rt = set_camera(setK, Ro, sett)

    #打印设定的相机内外参数
    print("Presets K=\n",np.round(K,2))
    print("Presets Rt=\n",np.round(Rt,2))

    # 计算像素坐标
    pixel_coor = compute_pixel(K, Rt, lines)
    #假设不知道相机的内外参数，只知道世界坐标系下的坐标和像素坐标系下的坐标，求解相机的内外参数。
    # 由于有12个未知数，所以需要至少6个点来求解，构造已知量矩阵
    A = np.zeros((len(lines) * 2, 12))
    for i in range(len(lines)):
        A[2 * i, :4] = lines[i]
        A[2 * i, 8:] = lines[i] * (-pixel_coor[i, 0])
        A[2 * i + 1, 4:8] = lines[i]
        A[2 * i + 1, 8:] = lines[i] * (-pixel_coor[i, 1])

    # 使用SVD分解来求解A的最小奇异值对应的列向量
    _, _, Vt = np.linalg.svd(A)
    # 取最小奇异值对应的列向量,变为3×4的矩阵形式
    M = Vt[-1,:].reshape(3, 4)

    #使用QR分解来求解M的内外参数，要和R，K的形式一致
    #K是一个上三角矩阵， R是一个正交矩阵
    #QR分解可以分解为一个正交矩阵和上三角矩阵的乘积
    #M=KR M^-1=(KR)^-1=R^-1K^-1 即R^-1K^-1=QR(M^-1)
    R,K=np.linalg.qr(np.linalg.inv(M[:,:3]))
    R=np.linalg.inv(R)
    K=np.linalg.inv(K)
    #如果K的第一个元素小于0，那么将K和R都乘以-1
    if (K / K[2, 2])[0,0]<0:
        K=np.dot(K,[[-1,0,0],[0,-1,0],[0,0,1]])
        R=np.dot([[-1,0,0],[0,-1,0],[0,0,1]],R)
    #求解t
    t = np.linalg.inv(K).dot( M[:, 3])
    K = K / K[2, 2] # 归一化

    # #漂亮的打印结果,四舍五入为
    print("Estimates K=\n",np.round(K,2))
    print("Estimates R=\n",np.round(R,2))
    print("Estimates t=\n",np.round(t,2))

    #可视化house.p3d
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(lines[:, 0], lines[:, 1], lines[:, 2], c='r', marker='o')
    ax.scatter(pixel_coor[:, 0], pixel_coor[:, 1], c='b', marker='*')
    #将其中的一部分点连线
    for i in range(len(lines)):
        if i % 20 == 0:
            ax.plot([lines[i, 0], pixel_coor[i, 0]], [lines[i, 1], pixel_coor[i, 1]], [lines[i, 2], 1], c='g')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #将相机内外参数K，R，t打印在窗口
    ax.text(-2, 2, -4, 'Estimates K=\n' + str(np.round(K, 2)))
    ax.text(-2, 2, 0, 'Estimates R=\n' + str(np.round(R, 2)))
    ax.text(-2, 2, 4, 'Estimates t=\n' + str(np.round(t, 2)))
    plt.show()