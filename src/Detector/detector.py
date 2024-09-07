"""
Script to run detection on an image and return bounding boxes in format (xyhw) such that it can then be tracked by tracker.
possible TODO:
- Consideration on improving fps (model optimization, use of more efficient models, portable hardware accelerator??)
- further roboust tracking: DeepSORT, multi-object tracking to track multiple objects
- Implement interface to allow for dynamic target selection
"""
import torch
from torchvision.models import detection
from torchvision.ops import box_convert
import time
import cv2
import numpy as np
import logging


# ssdlite320 with mobilenet_v3_large_weights box MAP (21.3), Params (3.4M), GFLOPs (0.58)
ssd_model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
ssd_model.eval()

# faster rcnn with mobilenet_v3_large_fpn_weights, box MAP (32.8), Params (19.4M), GFLOPs (4.49)
fasterrcnn_mobilenet = detection.fasterrcnn_mobilenet_v3_large_fpn(weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
fasterrcnn_mobilenet.eval()

# fcos resnet50 box MAP (39.2), Params (32.3M) and GFLOPs (128.21)
fcos_resnet50 = detection.fcos_resnet50_fpn(weights=detection.FCOS_ResNet50_FPN_Weights.DEFAULT)
fcos_resnet50.eval()

#  faster rcnn woth resnet 40, box MAP (41.5), Params (38.2M), GFLOPs (152.34)
retina_resnet50 = detection.retinanet_resnet50_fpn_v2(weights=detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
retina_resnet50.eval()


# frame: batched or single images (B, C, H, W) with images rescaled to 0.0, 1.0
def run_object_detection(frame: np.ndarray, model_name: str) -> dict[torch.Tensor]:
    # transpose the image to color first
    # do the image need to be resized to specific height & width?
    frame = frame.transpose((2, 0, 1))
    frame = frame / 255
    frame = np.expand_dims(frame, axis=0)
    torch_frame = torch.from_numpy(frame)

    start_time = time.time()
    match model_name:
        case "ssd_model":
            predictions = ssd_model(torch_frame.float())
        
        case "fasterrcnn_mobilenet":
            predictions = fasterrcnn_mobilenet(torch_frame.float())
        
        case "fcos_resnet50":
            predictions = fcos_resnet50(torch_frame.float())

        case "retina_resnet50":
            predictions = retina_resnet50(torch_frame.float())
            pass

        case _:
            print("Invalid model....")
            return -1

    print(f"============== FPS: {1.0 /(time.time() - start_time)} ==============")

    # dict with keys: box, scores, label
    # single batch
    return predictions[0]


# returned box in format of xyxy, tracker requires box in format xyhw
def convert_box_format(box_xyxy: np.ndarray) -> np.array:
    return box_convert(torch.from_numpy(box_xyxy), "xyxy", "xywh").detach().numpy()
