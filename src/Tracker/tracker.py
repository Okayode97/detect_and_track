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
import cv2
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

        # update count
        # temp solution to determine when to use prediction or estimation
        self.update_count = 0

        # staleness count
        # used to track how long the filter has not recieved measurement
        self.staleness_count = 0

    # predict
    def predict(self) -> None:
        self.filter.predict()

    # update
    def update(self, detection: np.ndarray) -> None:
        self.filter.predict()
        if detection.size != 0:
            self.update_count += 1
            self.update_count = min(25, self.update_count)
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

    def get_prediction(self):
        # returns prediction for next time step
        predicted_state_variable, _ = self.filter.get_prediction()

        estimated_position = np.array([predicted_state_variable[0],
                                       predicted_state_variable[2],
                                       predicted_state_variable[4],
                                       predicted_state_variable[6]])

        estimated_velocity = np.array([predicted_state_variable[1],
                                       predicted_state_variable[3],
                                       predicted_state_variable[5],
                                       predicted_state_variable[7]])
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

def draw_filters_box_estimates_onto_frame(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    frame_ = frame.copy()
    for track in tracks:
        x, y, h, w = track.filter.get_estimated_state()[0]
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(frame_, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame_, str(track.filter_id), (top_left[0]+50, top_left[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame_


class Tracker:
    def __init__(self):
        self.list_of_tracks: list[Track] = []
        self.track_id: int = 0
        self.dt = 1
        self.track_max_staleness = 4
        self.track_update_count_threshold = 20
    
    def associate_detections_to_tracks(self, detections: np.ndarray) -> dict[int, int]:
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
            # use filter prediction for next time step if it's recieved updates 20, if less use filters estimation
            # TODO: Ideally use error in state co-variance matrix to determine use of prediction
            track_bbox_prediction = track.filter.get_prediction()[0] if track.filter.update_count > self.track_update_count_threshold else track.filter.get_estimated_state()[0] 
            filter_bbox_centre = get_bbox_centre(track_bbox_prediction)
            filter_distances: np.array = np.array([])

            # get detections bbox centre
            for detection in detections:
                # TODO: Ideally switch to using IOU
                measurement_bbox_centre = get_bbox_centre(detection)
                filter_distances = np.append(filter_distances, np.linalg.norm(measurement_bbox_centre - filter_bbox_centre))

            # append the vectors together    
            cost_matrix = np.vstack([cost_matrix, filter_distances])

        # run linear sum assignment to get associations between tracks and centre
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        associations = {}
        for filter_id, associated_detection_id in zip(row_ind, col_ind):
            associations[int(filter_id)] = int(associated_detection_id)
        
        return associations


    # detections = np.ndarray
    def update_filters(self, detections: np.ndarray):
        
        # associate detections to tracks
        associations = self.associate_detections_to_tracks(detections)

        # for associated track, update it with the associated detection
        for filter_id, associated_detection_id in associations.items():
            self.list_of_tracks[filter_id].filter.update(detections[associated_detection_id])

            # decrease staleness count if it's greater than 0
            if self.list_of_tracks[filter_id].filter.staleness_count > 0:
                self.list_of_tracks[filter_id].filter.staleness_count-=1 
                self.list_of_tracks[filter_id].filter.staleness_count = max(0, self.list_of_tracks[filter_id].filter.staleness_count)            

        # if there are no associated track for a detection, update it with an empty detection
        for i, _ in enumerate(self.list_of_tracks):
            if i not in associations.keys():
                self.list_of_tracks[i].filter.update(np.array([]))

                # increase staleness count for unassociated track, maxed at traxk max staleness
                self.list_of_tracks[i].filter.staleness_count+=1 
                self.list_of_tracks[i].filter.staleness_count = min(self.track_max_staleness, self.list_of_tracks[i].filter.staleness_count)

                # remove track if it's staleness is at max value.
                if self.list_of_tracks[i].filter.staleness_count == self.track_max_staleness:
                    self.list_of_tracks.pop(i)


        # if there are no associated detection, create a new track for the detection
        for i, detection in enumerate(detections):
             if i not in associations.values():
                track_ = Track(KF_filter(detection, self.dt), self.track_id)
                self.list_of_tracks.append(track_)
                self.track_id += 1

    def set_filter_dt(self, dt):
        self.dt = dt
    