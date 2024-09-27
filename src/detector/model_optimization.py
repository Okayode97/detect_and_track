"""
Basic script to experiment with optimizing detection model to run at faster fps on a raspberry pi
Workflow
- Experiment with model optimizations methods
- Run quantized model live and log FPS, time taken, Optionally log detections.
    - further update to log metrics on Raspberry pi utilization
"""
import os
import time
import json

from label import coco_labels

import cv2
import numpy as np
import torch
from torchvision.models import detection


CAM = cv2.VideoCapture(0)

# load, compile and save the model
# ssd_model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
# ssd_model.eval()

# optimized_model = torch.compile(ssd_model)
# torch.save(optimized_model, "optimized_ssdlite320_mobilenet.pt")


# load the saved model
optimized_model = torch.load("optimized_ssdlite320_mobilenet.pt", weights_only=False)


def run_object_detection(model, frame) -> dict:
    # transpose the image to color first, reduce range to 0 - 1 and add batch dimension.
    frame = frame.transpose((2, 0, 1))
    frame = frame / 255
    frame = np.expand_dims(frame, axis=0)
    torch_frame = torch.from_numpy(frame).float()

    prediction = model(torch_frame)

    return prediction[0]


def run_on_single_frame_and_log_performance(model, model_name:str):
    log_filename = "logged_performance.json"
    logged_json_data = {}

    # load the data from the json file if it exists
    # if it doesn't exist create log file
    if os.path.exists(log_filename):
        with open(log_filename) as f:
            logged_json_data = json.load(f)
    else:
        with open(log_filename, "w") as f:
            json.dump({}, f, indent=4)
    
    if CAM.isOpened():
        ret, frame = CAM.read()

        # run image through the model
        start_time = time.time()
        predictions = run_object_detection(model, frame)
        end_time = time.time()

        # print out top three predictions
        scores = predictions["scores"].detach().numpy()
        labels = predictions["labels"].detach().numpy()

        top_three_detection_scores = np.sort(scores, axis=0)[-3:]
        for score in top_three_detection_scores:
            index = np.where(scores == score)[0][0]

            print(f"Detection: {coco_labels[labels[index]]}, Score: {score}")

        # log model performance
        elapsed_time = end_time - start_time
        fps = 1/elapsed_time

        print(f"Fps: {1/elapsed_time:.2f} | elapsed_time: {elapsed_time:.2f}")


def run_on_live_feed_and_log_performance(model, model_name: str):

    log_filename = "logged_performance.json"
    logged_json_data = {}
    logged_fps = []
    logged_elapsed_time = []

    # load the data from the json file if it exists
    # if it doesn't exist create log file
    if os.path.exists(log_filename):
        with open(log_filename) as f:
            logged_json_data = json.load(f)
    else:
        with open(log_filename, "w") as f:
            json.dump({}, f, indent=4)

    try:
        while CAM.isOpened():
            ret, frame = CAM.read()

            # run image through the model
            start_time = time.time()
            predictions = run_object_detection(model, frame)
            end_time = time.time()

            # print out top three predictions
            scores = predictions["scores"].detach().numpy()
            labels = predictions["labels"].detach().numpy()

            top_three_detection_scores = np.sort(scores, axis=0)[-3:]
            for score in top_three_detection_scores:
                index = np.where(scores == score)[0][0]

                print(f"Detection: {coco_labels[labels[index]]}, Score: {score}")

            # log model performance
            elapsed_time = end_time - start_time
            fps = 1/elapsed_time

            print(f"Fps: {1/elapsed_time:.2f} | elapsed_time: {elapsed_time:.2f}")

            logged_elapsed_time.append(elapsed_time)
            logged_fps.append(fps)

    except Exception:

        logged_json_data[model_name] = {"average_fps": sum(logged_fps)/len(logged_fps),
                                        "average_elapsed_time": sum(logged_elapsed_time)/len(logged_elapsed_time)}

        with open(log_filename, 'w') as f:
            json.dump(logged_json_data, f, indent=4)
    

# run_on_live_feed_and_log_performance(optimized_model, "compiled_ssd_model")
run_on_single_frame_and_log_performance(optimized_model, "compiled_ssd_model")