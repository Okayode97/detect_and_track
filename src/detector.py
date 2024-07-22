"""
Script to run detection on an image and return bounding boxes in format (xyhw) such that it can then be tracked by tracker.
"""
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torch
import time

model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
model.eval()
x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]

start_time = time.time()
predictions = model(x)
print("FPS: ", 1.0 / (time.time() - start_time))

print(f"{predictions=}")