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
import numpy as np
from typing import Optional


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
                           target_label_idx: Optional[int]=1) -> tuple[list]:
    
    # select n predictions with highest score
    scores = predictions["scores"].detach().numpy()
    bboxes = predictions["boxes"].detach().numpy()
    labels = predictions["labels"].detach().numpy()

    # selected detections
    top_n_labels = []
    top_n_bboxes = []
    top_n_scores = []

    # get top scores
    top_detection_scores = np.sort(scores, axis=0)[-n_detection:]

    # iterate through top detection & filter for target label
    for score in top_detection_scores:
        idx = np.where(scores == score)[0][0]

        if labels[idx] == target_label_idx:
            top_n_bboxes.append(bboxes[idx])
            top_n_scores.append(scores[idx])
            top_n_labels.append(labels[idx])

    return (top_n_scores, top_n_bboxes, top_n_labels)


def run_object_detection(model, frame: torch.Tensor) -> dict[torch.Tensor]:
    """
    Function runs object detection with single batched frame passed to the model.
    return is model's prediction
    """
    model.eval()
    with torch.no_grad:
        predictions = model(frame)

    return predictions[0]


def run_full_detection_pipeline(model, frame, n_detection):
    # preprocess the frame before passing it to the model
    torch_frame = preprocess_frame(frame)

    model_predictions = run_object_detection(model, torch_frame)

    return select_top_n_detection(model_predictions, n_detection)


# returned box in format of xyxy, tracker requires box in format xyhw
def convert_box_format(box_xyxy: np.ndarray) -> np.array:
    return box_convert(torch.from_numpy(box_xyxy), "xyxy", "xywh").detach().numpy()
