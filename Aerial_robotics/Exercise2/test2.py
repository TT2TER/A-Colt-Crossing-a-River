import numpy as np

def rodrigues_rotation_matrix(u, a):
    # Normalize the rotation axis
    u = u / np.linalg.norm(u)
    
    # Calculate the cross product matrix of the rotation axis
    u_cross = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
    
    # Calculate the rotation matrix using Rodrigues' formula
    R = np.eye(3) + np.sin(a) * u_cross + (1 - np.cos(a)) * np.dot(u_cross, u_cross)
    
    return R

# Example usage
u = np.array([0.577, 0.577, 0.577])  # Rotation axis（实际上应该是转置）
a = 2*np.pi / 3  # Rotation angle (in radians)

rotation_matrix = rodrigues_rotation_matrix(u, a)
print(rotation_matrix)

# 2.3.2 旋转矩阵转换为旋转向量
# Path: test3.py
# import numpy as np
