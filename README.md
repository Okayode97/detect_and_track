# detect_and_track
repo for detector and tracker to try out tracking on a mobile robot

## tests
- `python -m pytest tests\`


## Main app
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
So far nothing seems to be able to run decently on the PI, Optimization methods that have been tried
- quantizing + compiling the model
with both method the pi still crashes and doesn't respond making working on it remotely very difficult, i've opted to run a model server on a different laptop to handle the running the model and simply return the detection back to the raspberry pi.


## Areas in need of improvements
- Target re-identification. Target track are occasionally dropped and re-identified with a different id. It would be ideal for the tracker to maintain consistent id for given target (deepsort resolve this...) 
- Requires a good balance on detector inference speed and live performance.
    - Update (12/10/2024): Moved detector from running on device to running on a local server. This offsets computational cost in running model locally on raspberry pi. This change has required that the raspberry pi now acts as a client which sends frames to a local linux server on the same network. So now we can get detection from what raspberry pi camera sees, but the model is stil not particually fast or accurate.


## Work to do
- Interface to select single target when multiple targets are in view of camera
- logging to record performance of full system together
  - model: fps, detections, comparsion across different model
  - tracker: Number of active tracks & id, dropped tracks & re-id tracks
  - server: server latency
- model optimization
- interface to control vehicle
- functionality to convert track into information on how to move vehicle
- implement simple logic to get track re-identification
- breakdown server and main app into components and deploy to local kubernetes cluster

Further extension on work to be done.   
- experiment with YOLO
- experiment with deepsort