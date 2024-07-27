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

# optimize for raspberry pi
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
    # box format?? xyxy
    return predictions[0]


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

        if not ret:
            print("Unable to read frame from camera...")
            break

        predictions = run_object_detection(frame)
        scores = predictions["scores"].detach().numpy()
        bbox = predictions["boxes"].detach().numpy()
        labels = predictions["labels"].detach().numpy()

        # get the top 3 prediction
        top_three_detection_scores: np.array = np.sort(scores, axis=0)[-3:]
        top_three_bbox = []
        top_three_labels = []

        for score in top_three_detection_scores:
            index = np.where(scores == score)[0][0]
            top_three_bbox.append(bbox[index])
            top_three_labels.append(labels[index])

        for detection in top_three_bbox:
            print(detection)
            cv2.rectangle(frame, (int(detection[0]), int(detection[1])),
                                 (int(detection[2]), int(detection[3])), (0, 255, 0), 1)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) == ord('q'):
            break



if __name__ == "__main__":
    run_inference_on_live_image()
