"""
Script to run detection on an image and return bounding boxes in format (xyhw) such that it can then be tracked by tracker.
possible TODO:
- Consideration on improving fps (model optimization, use of more efficient models, portable hardware accelerator??)
- further roboust tracking: DeepSORT, multi-object tracking to track multiple objects
- Implement interface to allow for dynamic target selection
"""
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.ops import box_convert
import time
import cv2
import numpy as np
import logging
from label import coco_labels


IMAGE_DIM = 320

# optimize for raspberry pi
ssd_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
ssd_model.eval()


# frame: batched or single images (B, C, H, W) with images rescaled to 0.0, 1.0
def run_object_detection(frame: np.ndarray) -> dict[torch.Tensor]:
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
    # single batch
    return predictions[0]

# returned box in format of xyxy, tracker requires box in format xyhw
def convert_box_format(box_xyxy: np.ndarray) -> np.array:
    return box_convert(torch.from_numpy(box_xyxy), "xyxy", "xywh").detach().numpy()

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
        top_three_bbox: list[int] = []
        top_three_labels: list[int] = []

        for score in top_three_detection_scores:
            index = np.where(scores == score)[0][0]
            top_three_bbox.append(bbox[index])
            top_three_labels.append(labels[index])

        for detection, label_index in zip(top_three_bbox, top_three_labels):
            cv2.rectangle(frame, (int(detection[0]), int(detection[1])),
                                 (int(detection[2]), int(detection[3])), (0, 255, 0), 1)
            cv2.putText(frame, coco_labels[label_index], (int(detection[0])+50, int(detection[1])+50), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (0, 255, 0), 2, cv2.LINE_AA) 

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) == ord('q'):
            break



if __name__ == "__main__":
    run_inference_on_live_image()
