"""
Script to run detection on an image and return bounding boxes in format (xyhw) such that it can then be tracked by tracker.
"""
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import time
import cv2
import numpy as np
import logging


list_of_captured_image = []
ssd_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
ssd_model.eval()

def run_object_detection(frame: torch.tensor):
    predictions = ssd_model(frame)
    print(f"Prediction: {predictions}")
    pass

def capture_image():
    CAM = cv2.VideoCapture(0)

    if CAM.isOpened():
        print("Camera is opened")
    else:
        print("Unable to open camera")

    while CAM.isOpened():
        ret, frame = CAM.read()

        if not ret:
            print("Unable to read frame from camera...")
            break

        cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            # list_of_captured_image.append(frame)
            print("Captured image...")
            color_first = np.moveaxis(frame, -1, 0)
            color_first = np.expand_dims(color_first, 0)
            print(color_first.shape)
            torch_frame = torch.from_numpy(color_first)
            print(torch_frame.shape)
            run_object_detection(torch_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    capture_image()
