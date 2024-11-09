"""
Approach to model server
- main benefit is to offload computational load of running model from raspberry pi to another machine
- Send frames from main client (Raspberry pi) to a server (Old linux laptop)
- Server runs object detection using selected model, results from server is then sent back to client
- client recieves 
"""
from fastapi import FastAPI, Request
import cv2
import numpy as np
import time
from detector.detector import ssd_model,  run_full_detection, ModelQuantizationWrapper, quantize_model_with_backend, find_all_conv2d_norm_activation_blocks_and_fuse_them
from detector.custom_logging import log_results, log_detections

app = FastAPI()
find_all_conv2d_norm_activation_blocks_and_fuse_them(ssd_model.backbone.features)
ssd_model_wrapped_input = ModelQuantizationWrapper(ssd_model)
ssd_model_quantized = quantize_model_with_backend(ssd_model_wrapped_input)

# setup server
app.num_detections = 0
app.model = ssd_model_quantized
app.model_name = "ssd_model_quantized_test_2"
app.last_time = time.time()
app.capture_interval = 30
app.log_data = True

def decode_bytes_to_img(bytes):
    img = np.asarray(bytearray(bytes), dtype="uint8")
    decoded_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return decoded_img


@app.post("/images")
async def get_model_prediction(request: Request):
    data = await request.body()
    img = decode_bytes_to_img(data)
    detections = run_full_detection(app.model, img, 5)

    if app.log_data:
        log_results(detections["metrics"], "baseline", app.model_name)
        current_time = time.time()
        if current_time - app.last_time >= app.capture_interval:
            log_detections(img, detections["detections"], f"./.data/{app.model_name}_detection_{app.num_detections}.png")
            app.num_detections += 1
            app.last_time = current_time
    return detections

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
