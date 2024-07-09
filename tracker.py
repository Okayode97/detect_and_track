"""
Script to contain filters and trackers to track bounding box returned by a detector.
Resources:
- Kalman-and-Bayesian-Filters-in-Python/08-Designing-Kalman-Filters.ipynb
- https://thekalmanfilter.com/kalman-filter-explained-simply/
"""
from typing import Optional
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

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
    def __init__(self):
        pass

class Tracker:
    def __init__(self):
        self.trackers: list[KF_filter] = []
    
    pass
