import numpy as np
from filterpy.kalman import KalmanFilter

def create_kalman_filter(cx, cy, w, h):
    kf = KalmanFilter(dim_x=6, dim_z=4)
    dt = 1.0  # frame time step

    kf.F = np.array([
        [1,0,0,0,dt,0],
        [0,1,0,0,0,dt],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ])
    kf.H = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0]
    ])
    kf.P *= 10
    kf.R *= 1
    kf.Q *= 0.01
    kf.x[:4] = np.array([cx, cy, w, h]).reshape(4,1)
    return kf
