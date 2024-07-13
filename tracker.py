"""
Script to contain filters and trackers to track bounding box returned by a detector.
Resources:
- Kalman-and-Bayesian-Filters-in-Python/08-Designing-Kalman-Filters.ipynb
- https://thekalmanfilter.com/kalman-filter-explained-simply/
- https://github.com/abewley/sort/blob/master/sort.py
- https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
"""
from typing import Optional
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


@dataclass
class Detection:
    # expected format is xyhw
    bbox: Optional[np.ndarray] = None # x, y, h, w

    def convert_xyxy_to_xyhw(self, xyxy: np.ndarray):
        height = xyxy[2] - xyxy[0]
        width = xyxy[-1] - xyxy[1]
        self.bbox = np.array([xyxy[0], xyxy[1], height, width])
    
    def convert_xyhw_to_xyxy(self):
        if self.bbox is not None:
            return np.array([self.bbox[0], self.bbox[1],
                             self.bbox[0]+self.bbox[2], self.bbox[1]+self.bbox[3]])

class KF_filter:
    def __init__(self, dt):
        # state variables: x, x_hat, y, y_hat, h, h_hat,  w, w_hat
        # measurement: x, y, h, w
        self.filter = KalmanFilter(dim_x=8, dim_z=4)

        # define the state transition matrix
        # converts our state variable to the state variable in the next time step 
        self.filter.F = np.array([[1, dt, 0, 0, 0, 0, 0, 0], # x
                                  [0, 1, 0, 0, 0, 0, 0, 0],  # x_hat
                                  [0, 0, 1, dt, 0, 0, 0, 0], # y
                                  [0, 0, 0, 1, 0, 0, 0, 0],  # y_hat
                                  [0, 0, 0, 0, 1, dt, 0, 0], # h
                                  [0, 0, 0, 0, 0, 1, 0, 0],  # h_hat
                                  [0, 0, 0, 0, 0, 0, 1, dt], # w
                                  [0, 0, 0, 0, 0, 0, 0, 1]   # w_hat
                                  ])

        # define the process noise matrix
        # assume noise is discrete time wiener process and is constant for each time period.
        # Assumption allows user to define variance (how much we believe model changes between steps)
        # Another assumption is that the noise in x, y, h, w are independent so covariance between them shoild be zero
        q = Q_discrete_white_noise(dim=2, dt=dt, var=1)
        self.filter.Q = block_diag(q, q, q, q)

        # define control function.
        self.filter.B = 0

        # define the measurement function
        # converts state variables to measurement variables using the H matrix
        self.filter.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0]])

        # define the measurement covariance matrix
        # similar to the state variable.
        # it is assume that the noise in the measurements are independent from each other
        self.filter.R = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        # define initial state
        self.filter.x = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        # define the initial state covariance matrix
        self.filter.P *=1e+2

    # predict
    def predict(self) -> None:
        self.filter.predict()

    # update
    def update(self, detection: np.ndarray) -> None:
        self.filter.update(detection)

    # get current estimated state
    def get_estimated_state(self) -> tuple[np.array]:
        # state variables: x, x_hat, y, y_hat, h, h_hat,  w, w_hat

        estimated_position = np.array([self.filter.x[0],
                                       self.filter.x[2],
                                       self.filter.x[4],
                                       self.filter.x[6]])

        estimated_velocity = np.array([self.filter.x[1],
                                       self.filter.x[3],
                                       self.filter.x[5],
                                       self.filter.x[7]])
        return (estimated_position, estimated_velocity)


class Tracker:
    def __init__(self):
        self.trackers: list[KF_filter] = []
    
    def associate(self):
        pass

    def update_tracks(self):
        pass
    
    pass
