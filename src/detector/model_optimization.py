""""""
# torchvision.transforms for Image Preprocessing
# torch.quantization.quantize_dynamic() reducdes model size
# Used a single def fr log_performance
"""""""



import os
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import detection
import psutil  # For Raspberry Pi resource monitoring

from label import coco_labels


CAM = cv2.VideoCapture(0)
transform = T.Compose([
    T.ToTensor()  # Converts to Tensor and normalizes between [0, 1]
])


def run_object_detection(model, frame) -> dict:
    # Transform the image to tensor
    torch_frame = transform(frame).unsqueeze(0)
    prediction = model(torch_frame)
    return prediction[0]


def log_performance_metrics(log_filename, model_name, fps_list, elapsed_time_list):
    logged_data = {}

    # Load existing data
    if os.path.exists(log_filename):
        with open(log_filename, 'r') as f:
            logged_data = json.load(f)
    
    # Calculate average metrics
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    avg_elapsed_time = sum(elapsed_time_list) / len(elapsed_time_list) if elapsed_time_list else 0

    # Log the new data
    logged_data[model_name] = {
        "average_fps": avg_fps,
        "average_elapsed_time": avg_elapsed_time
    }

    with open(log_filename, 'w') as f:
        json.dump(logged_data, f, indent=4)


def run_on_single_frame_and_log_performance(model, model_name: str):
    log_filename = "logged_performance.json"
    
    if CAM.isOpened():
        ret, frame = CAM.read()
        if not ret:
            print("Failed to grab frame.")
            return

        # Run image through the model
        start_time = time.time()
        predictions = run_object_detection(model, frame)
        end_time = time.time()

        # Print out top three predictions
        scores = predictions["scores"].detach().numpy()
        labels = predictions["labels"].detach().numpy()
        top_indices = np.argsort(scores)[-3:][::-1]  # Get top 3 indices in descending order

        for idx in top_indices:
            print(f"Detection: {coco_labels[labels[idx]]}, Score: {scores[idx]:.2f}")

        # Log model performance
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time
        print(f"FPS: {fps:.2f} | Elapsed Time: {elapsed_time:.2f} seconds")

        log_performance_metrics(log_filename, model_name, [fps], [elapsed_time])


def run_on_live_feed_and_log_performance(model, model_name: str):
    log_filename = "logged_performance.json"
    logged_fps = []
    logged_elapsed_time = []

    try:
        while CAM.isOpened():
            ret, frame = CAM.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Run image through the model
            start_time = time.time()
            predictions = run_object_detection(model, frame)
            end_time = time.time()

            # Print out top three predictions
            scores = predictions["scores"].detach().numpy()
            labels = predictions["labels"].detach().numpy()
            top_indices = np.argsort(scores)[-3:][::-1]  # Get top 3 indices in descending order

            for idx in top_indices:
                print(f"Detection: {coco_labels[labels[idx]]}, Score: {scores[idx]:.2f}")

            # Log model performance
            elapsed_time = end_time - start_time
            fps = 1 / elapsed_time
            print(f"FPS: {fps:.2f} | Elapsed Time: {elapsed_time:.2f} seconds")

            logged_elapsed_time.append(elapsed_time)
            logged_fps.append(fps)

            # Log Raspberry Pi CPU and Memory usage
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            print(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_info.percent}%")

    except KeyboardInterrupt:
        print("Stopping live feed...")

    finally:
        log_performance_metrics(log_filename, model_name, logged_fps, logged_elapsed_time)
        CAM.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the model
    ssd_model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    ssd_model.eval()

    # Quantize the model (dynamic quantization)
    quantized_model = torch.quantization.quantize_dynamic(ssd_model, {nn.Linear}, dtype=torch.qint8)

    # Compile the model for faster inference (PyTorch 2.x)
    if hasattr(torch, 'compile'):
        optimized_model = torch.compile(quantized_model)
    else:
        optimized_model = quantized_model

    torch.save(optimized_model, "quantized_and_compiled_ssdlite320_mobilenet.pt")

    # Load the saved model
    optimized_model = torch.load("quantized_and_compiled_ssdlite320_mobilenet.pt", weights_only=False)

    # Run the model on live feed and log performance
    run_on_live_feed_and_log_performance(optimized_model, "compiled_ssd_model")
    # or run on a single frame
    # run_on_single_frame_and_log_performance(optimized_model, "compiled_ssd_model")



# load the saved model
optimized_model = torch.load("quantized_and_compiled_ssdlite320_mobilenet.pt", weights_only=False)

# run_on_live_feed_and_log_performance(optimized_model, "compiled_ssd_model")
run_on_single_frame_and_log_performance(optimized_model, "compiled_ssd_model")
