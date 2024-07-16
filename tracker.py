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
from scipy.optimize import linear_sum_assignment


class KF_filter:
    def __init__(self, detection: np.ndarray, dt: float):
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
        # populate the initial state using the measurement, set the velocity to be 0 
        self.filter.x = np.array([(int(x), 0) for x in detection]).reshape(-1, 8)[0]

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


@dataclass
class Track:
    # expected format is xyhw
    filter: KF_filter # x, y, h, w
    filter_id: int

# bbox (np.ndarray): x, y, h, w
# note x linked to w and not h, vice-versa with y
def get_bbox_centre(bbox: np.ndarray) -> np.ndarray:
    return np.array([bbox[0] + bbox[3]/2, bbox[1] + bbox[2]/2])

class Tracker:
    def __init__(self):
        self.list_of_tracks: list[Track] = []
    
    def associate_detections_to_tracks(self, detections: np.ndarray):
        """
        associate detections to existing tracks. Would return an empty array if there is no
        track to associate detections with.
        Args:
            - detections: np.ndarray, (N, x, y, h, w)
        """
        
        # define a matrix composed of distance between estimated bbox & actual bbox
        cost_matrix: np.array = np.empty((0, len(detections)))

        # get the estimated bbox centre
        for track in self.list_of_tracks:
            filter_bbox_centre = get_bbox_centre(track.filter.get_estimated_state()[0])
            filter_distances: np.array = np.array([])

            # get detections bbox centre
            for detection in detections:
                measurement_bbox_centre = get_bbox_centre(detection)
                filter_distances = np.append(filter_distances, np.linalg.norm(measurement_bbox_centre - filter_bbox_centre), axis=0)

            # append the vectors together    
            cost_matrix = np.vstack([cost_matrix, filter_distances])

        # run linear sum assignment to get associations between tracks and centre
        row_ind, _ = linear_sum_assignment(cost_matrix)
        
        # map row to col (track to detection)
        return row_ind


    # detections = np.ndarray
    def update_filters(self, detections: np.ndarray):
        
        # associate detections to tracks
        associations = self.associate_detections_to_tracks(detections)

        # for associated detections, update it with the associated detection

        # if there are no associated detection, create a new track for the detection

        # if there are no associated track for a detection,
        # - update it with empty detection, but increase it's staleness, 
        # - ultimately remove it if it's staleness goes beyong a certain point

    
        pass
