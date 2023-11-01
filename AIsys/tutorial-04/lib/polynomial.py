import numpy as np

'''
五次多项式
'''
class QuinticPolynomial:
    def __init__(
        self,
        p_start     : float,   # 起点的函数值 p(t)
        p_dot_start : float,   # 起点函数值的导数 p'(t)
        p_ddot_start: float,   # 起点函数值的二阶导 p''(t)
        p_end       : float,   # 终点的函数值 p(t)
        p_dot_end   : float,   # 终点函数值的导数 p'(t)
        p_ddot_end  : float,   # 终点函数值的二阶导 p''(t)
        sample_time : np.array # 离散的时间
        ):

        t_end = sample_time[-1]

        matrix_a = np.mat([[1, 0    , 0         , 0             , 0              , 0              ],
                           [0, 1    , 0         , 0             , 0              , 0              ],
                           [0, 0    , 2         , 0             , 0              , 0              ],
                           [1, t_end, t_end ** 2, t_end ** 3    , t_end ** 4     , t_end ** 5     ],
                           [0, 1    , 2 * t_end , 3 * t_end ** 2, 4 * t_end ** 3 , 5 * t_end ** 4 ],
                           [0, 0    , 2         , 6 * t_end     , 12 * t_end ** 2, 20 * t_end ** 3]])
        
        vector_b = np.mat([p_start, p_dot_start, p_ddot_start, p_end, p_dot_end, p_ddot_end]).T

        # calculate coefficients of quintic polynomial
        coeff_c = np.linalg.solve(matrix_a, vector_b)

        # d and temporal derivatives
        self.p_ = np.mat(coeff_c[0]
                         + coeff_c[1] * sample_time
                         + coeff_c[2] * sample_time ** 2
                         + coeff_c[3] * sample_time ** 3
                         + coeff_c[4] * sample_time ** 4
                         + coeff_c[5] * sample_time ** 5)
        
        self.p_dot_ = np.mat(coeff_c[1]
                            + 2 * coeff_c[2] * sample_time
                            + 3 * coeff_c[3] * sample_time ** 2
                            + 4 * coeff_c[4] * sample_time ** 3
                            + 5 * coeff_c[5] * sample_time ** 4)
        
        self.p_ddot_ = np.mat(2 * coeff_c[2]
                             + 6  * coeff_c[3] * sample_time
                             + 12 * coeff_c[4] * sample_time ** 2
                             + 20 * coeff_c[5] * sample_time ** 3)
        
        self.p_dddot_ = np.mat(6 * coeff_c[3] 
                              + 24 * coeff_c[4] * sample_time 
                              + 60 * coeff_c[5] * sample_time ** 2)
        
class QuarticPolinomial:
    def __init__(
        self,
        p_start     : float, # 起点的函数值 p(t)
        p_dot_start : float, # 起点的函数导数 p'(t)
        p_ddot_start: float, # 起点得到函数二阶导 p''(t)
        p_dot_end   : float, # 终点的函数导数 p'(t)
        p_ddot_end  : float, # 终点函数的二阶导 p''(t)
        sample_time : np.array
        ):

        t_end = sample_time[-1]

        matrix_a = np.mat(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 1, 2 * t_end, 3 * t_end ** 2, 4 * t_end ** 3],
                [0, 0, 2, 6 * t_end, 12 * t_end ** 2],
            ]
        )
        vector_b = np.mat([p_start, p_dot_start, p_ddot_start, p_dot_end, p_ddot_end]).T

        # 计算四次多项式的参数
        coeff_c = np.linalg.solve(matrix_a, vector_b)

        # s and temporal derivatives
        self.p_ = np.mat(
            coeff_c[0]
            + coeff_c[1] * sample_time
            + coeff_c[2] * sample_time ** 2
            + coeff_c[3] * sample_time ** 3
            + coeff_c[4] * sample_time ** 4
        ).T
        self.p_dot_ = np.mat(
            coeff_c[1]
            + 2 * coeff_c[2] * sample_time
            + 3 * coeff_c[3] * sample_time ** 2
            + 4 * coeff_c[4] * sample_time ** 3
        ).T
        self.p_ddot_  = np.mat(2 * coeff_c[2] + 6 * coeff_c[3] * sample_time + 12 * coeff_c[4] * sample_time ** 2).T
        self.p_dddot_ = np.mat(6 * coeff_c[3] + 24 * coeff_c[4] * sample_time).T