"""
Basic script to experiment with optimizing detection model to run at faster fps on a raspberry pi
Workflow
- Experiment with model optimizations methods
- Run quantized model live and log FPS, time taken, Optionally log detections.
    - further update to log metrics on Raspberry pi utilization

Nothing seems to work...
"""
import torch
import torch.nn as nn
from torchvision.models import detection



# load the model
ssd_model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
ssd_model.eval()


# quantize the model
quantized_model = torch.quantization.quantize_dynamic(ssd_model, {nn.Linear}, dtype=torch.qint8)

# compile the model
optimized_model = torch.compile(quantized_model)
torch.save(optimized_model, "quantized_and_compiled_ssdlite320_mobilenet.pt")


# # load the saved model
# optimized_model = torch.load("quantized_and_compiled_ssdlite320_mobilenet.pt", weights_only=False)

# # run_on_live_feed_and_log_performance(optimized_model, "compiled_ssd_model")
# run_on_single_frame_and_log_performance(optimized_model, "compiled_ssd_model")