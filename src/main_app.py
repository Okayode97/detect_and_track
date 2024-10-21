from tracker.tracker import Tracker, draw_filters_box_estimates_onto_frame
from detector.detector import convert_box_format, run_full_detection, ssd_model

import time
import cv2
import numpy as np

def run_inference_on_live_image(headless: bool = False):

    # instantiate video capture
    CAM = cv2.VideoCapture(1)

    tracker_ = Tracker()

    start_time = None
    dt = 1

    if CAM.isOpened():
        print("Camera is opened")
    else:
        print("Unable to open camera")

    while CAM.isOpened():
        ret, frame = CAM.read()

        if not ret:
            print("Unable to read frame from camera...")
            break
        
        detections = run_full_detection(ssd_model, frame, 2)
        top_bbox = [detections["detections"][key]["boxes"] for key in detections["detections"].keys()] 

        if len(top_bbox) != 0:
            detections = np.vstack([convert_box_format(np.array(bbox)) for bbox in top_bbox])
        else:
            detections = np.array([])

        if start_time is not None:
            dt = time.time() - start_time
        tracker_.update_filters(detections)
        tracker_.set_track_dt(dt)
        start_time = time.time()


       # draw estimated state from previous time step before updating
        frame = draw_filters_box_estimates_onto_frame(frame, tracker_.list_of_tracks)

        if not headless:
            cv2.imshow("Image", frame)
            if cv2.waitKey(1) == ord('q'):
                break

run_inference_on_live_image()