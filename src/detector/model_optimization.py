"""
Basic script to experiment with optimizing detection model to run at faster fps on a raspberry pi
Workflow
- Experiment with model optimizations methods
- Run quantized model live and log FPS, time taken, Optionally log detections.
"""
import cv2
import os
import time
import json


CAM = cv2.VideoCapture(0)

def run_object_detection(model, frame):
    model(frame)
    pass

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

            if not ret:
                print("Unable to read frame from camera...")
                break

            # run image through the model
            start_time = time.time()
            run_object_detection(model, frame)
            end_time = time.time()

            # log model performance
            elapsed_time = end_time - start_time
            fps = 1/elapsed_time

            print(f"Fps: {1/elapsed_time:.2f} | elapsed_time: {elapsed_time:.2f}")

            logged_elapsed_time.append(elapsed_time)
            logged_fps.append(fps)

    except KeyboardInterrupt:

        logged_json_data[model_name] = {"average_fps": sum(logged_fps)/len(logged_fps),
                                        "average_elapsed_time": sum(logged_elapsed_time)/len(logged_elapsed_time)}

        with open(log_filename, 'w') as f:
            json.dump(logged_json_data, f, indent=4)
    

x = lambda x: x
run_on_live_feed_and_log_performance(x, "dummy_model")
