from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

from src.Tracker.tracker import Tracker
from src.Detector.detector import run_object_detection, convert_box_format
from src.Detector.label import coco_labels

import cv2
import numpy as np

def run_inference_on_live_image():

    # instantiate video capture
    CAM = cv2.VideoCapture(0)

    # optimize for raspberry pi
    ssd_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    ssd_model.eval()

    tracker_ = Tracker()

    if CAM.isOpened():
        print("Camera is opened")
    else:
        print("Unable to open camera")

    while CAM.isOpened():
        ret, frame = CAM.read()

        if not ret:
            print("Unable to read frame from camera...")
            break

        predictions = run_object_detection(frame, ssd_model)
        scores = predictions["scores"].detach().numpy()
        bbox = predictions["boxes"].detach().numpy()
        labels = predictions["labels"].detach().numpy()

        # get the top 3 prediction
        top_three_detection_scores: np.array = np.sort(scores, axis=0)[-3:]
        top_bbox: list[int] = []
        top_labels: list[int] = []

        for score in top_three_detection_scores:
            index = np.where(scores == score)[0][0]

            # filter to track people
            if labels[index] == 1:
                top_bbox.append(bbox[index])
                top_labels.append(labels[index])

        if len(top_bbox) != 0:
            detections = np.vstack([convert_box_format(bbox) for bbox in top_bbox])
        else:
            detections = np.array([])

        tracker_.update_filters(detections)
   
       # draw estimated state from previous time step before updating
        frame = tracker_.draw_filters_box_estimates_onto_frame(frame)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

run_inference_on_live_image()