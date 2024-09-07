# detect_and_track
repo for detector and tracker to try out tracking on a mobile robot

# tests
- `python -m pytest tests\`

# Main app
Runs the object detector on incoming frames and filters incoming detection for specific target before passing it to the tracker & tracks it.
- `python -m src\main_app.py`


## Notes
- initial detector models selected for target detections
- [Pretrained object detector](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection) models on pytorch.
  - SSDLite320_MobileNet_V3_Large_Weights, Box MAP (21.3), Params (3.4M), GFLOPs (0.58)
  - FasterRCNN_Mobilenet_V3_Large_FPN_Weights, Box Map (32.8), Params (19.4M), GFLOPs (4.49)
  - FCOS_ResNet50_FPN_Weights, Box Map (39.2), Params (39.2M), GFLOPs (128.21)
  - FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, Box Map (41.5), Params (38.2M), GFLOPs (152.24)


Models selected for balance between accuracy and performance. I've excluded FasterRCNN_ResNet50_FPN from the above as though it is reported to perform well, it requires much higher number of operations and as a baseline was extermely slow.

## Areas in need of improvements
- Target re-identification. Target track are occasionally dropped and re-identified with a different id. It would be ideal for the tracker to maintain consistent id for given target (deepsort resolve this...) 
- Requires a good balance on detector inference speed and live performance

## Work to do
- Interface to select single target when multiple targets are in view of camera
- logging to record performance of full system together
  - run time
  - model inference time, fps, target detections
  - tracker number of active tracks & id, dropped tracks
- functionality to convert track into information on how to move vehicle
- experiment with YOLO look into how to 
- experiment with deepsort