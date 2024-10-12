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
from detector.detector import retina_resnet50, ssd_model, run_full_detection

app = FastAPI()

def decode_bytes_to_img(bytes):
    img = np.asarray(bytearray(bytes), dtype="uint8")
    decoded_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return decoded_img


@app.get("/")
async def root():
    return {"msg": "hello world"}

@app.post("/images")
async def get_model_prediction(request: Request):
    data = await request.body()
    img = decode_bytes_to_img(data)
    detections = run_full_detection(ssd_model, img, 5)
    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
