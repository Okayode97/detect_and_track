"""
Script to run detection on an image and return bounding boxes in format (xyhw) such that it can then be tracked by tracker.
possible TODO:
- Consideration on improving fps (model optimization, use of more efficient models, portable hardware accelerator??)
- further roboust tracking: DeepSORT, multi-object tracking to track multiple objects
- Implement interface to allow for dynamic target selection
"""
import torch
from torch.nn import Module
from torchvision.models import detection
from torchvision.ops import box_convert
import numpy as np
from typing import Optional
from time import time


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


def preprocess_frame(frame: np.ndarray) -> torch.tensor:
    # transpose the image to color first, reduce it's range and add batch dimension
    frame = frame.transpose((2, 0, 1))
    frame = frame / 255
    frame = np.expand_dims(frame, axis=0)
    return torch.from_numpy(frame).float()


def select_top_n_detection(predictions: dict[torch.Tensor], n_detection: int,
                           target_label_idx: Optional[int]=1) -> list[dict]:
    
    # select n predictions with highest score
    scores = predictions["scores"].detach().numpy()
    bboxes = predictions["boxes"].detach().numpy()
    labels = predictions["labels"].detach().numpy()

    # selected detections
    selected_detections = []

    # get top scores
    top_detection_scores = np.sort(scores, axis=0)[-n_detection:]

    # iterate through top detection & filter for target label
    for score in top_detection_scores:
        idx = np.where(scores == score)[0][0]

        data_dict = {}
        if labels[idx] == target_label_idx:
            data_dict["score"] = scores[idx]
            data_dict["label"] = labels[idx]
            data_dict["boxes"] = bboxes[idx]
            selected_detections.append(data_dict)

    return selected_detections


def run_object_detection(model: Module, frame: torch.Tensor) -> dict[torch.Tensor]:
    """
    Function runs object detection with single batched frame passed to the model.
    return is model's prediction
    """
    model.eval()
    with torch.no_grad:
        predictions = model(frame)

    return predictions[0]


def run_full_detection(model: Module, frame: np.ndarray, n_detection: int) -> dict:
    # preprocess the frame before passing it to the model
    torch_frame = preprocess_frame(frame)

    start_time = time()
    model_predictions = run_object_detection(model, torch_frame)
    end_time = time()

    log_metrics = {"elasped_time": end_time - start_time, "fps": 1/(end_time-start_time)}

    selected_detections = select_top_n_detection(model_predictions, n_detection)

    # merge selected detections with logged_metrics
    results = {f"detection_{idx}": dict for idx, dict in enumerate(selected_detections)}
    
    # update selected detections with log metrics
    results.update(log_metrics)
    return results


# returned box in format of xyxy, tracker requires box in format xyhw
def convert_box_format(box_xyxy: np.ndarray) -> np.array:
    return box_convert(torch.from_numpy(box_xyxy), "xyxy", "xywh").detach().numpy()
