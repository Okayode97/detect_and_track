"""
Script to run detection on an image and return bounding boxes in format (xyhw) such that it can then be tracked by tracker.
"""
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision import transforms
import time
import cv2
import numpy as np
import logging


IMAGE_DIM = 320

list_of_captured_image = []
ssd_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
ssd_model.eval()

# frame: batched or single images (B, C, H, W) with images rescaled to 0.0, 1.0
def run_object_detection(frame: np.ndarray):
    # transpose the image to color first
    # do the image need to be resized to specific height & width?
    frame = frame.transpose((2, 0, 1))
    frame = frame / 255
    frame = np.expand_dims(frame, axis=0)
    torch_frame = torch.from_numpy(frame)

    start_time = time.time()
    predictions = ssd_model(torch_frame.float())
    print(f"============== FPS: {1.0 /(time.time() - start_time)} ==============")

    # dict with keys: box, scores, label
    # box format??
    return predictions
    pass

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms
])

def run_inference_on_live_image():

    # instantiate video capture
    CAM = cv2.VideoCapture(0)

    # set frame width, height & fps

    if CAM.isOpened():
        print("Camera is opened")
    else:
        print("Unable to open camera")

    while CAM.isOpened():
        ret, frame = CAM.read()

        run_object_detection(frame)
        
        if not ret:
            print("Unable to read frame from camera...")
            break



if __name__ == "__main__":
    run_inference_on_live_image()
