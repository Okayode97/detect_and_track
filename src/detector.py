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


# optimize for raspberry pi
ssd_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
ssd_model.eval()


# frame: batched or single images (B, C, H, W) with images rescaled to 0.0, 1.0
def run_object_detection(frame: np.ndarray, model) -> dict[torch.Tensor]:
    # transpose the image to color first
    # do the image need to be resized to specific height & width?
    frame = frame.transpose((2, 0, 1))
    frame = frame / 255
    frame = np.expand_dims(frame, axis=0)
    torch_frame = torch.from_numpy(frame)

    start_time = time.time()
    predictions = model(torch_frame.float())
    print(f"============== FPS: {1.0 /(time.time() - start_time)} ==============")

    # dict with keys: box, scores, label
    # single batch
    return predictions[0]

# returned box in format of xyxy, tracker requires box in format xyhw
def convert_box_format(box_xyxy: np.ndarray) -> np.array:
    return box_convert(torch.from_numpy(box_xyxy), "xyxy", "xywh").detach().numpy()
