"""
Approach to model server
- main benefit is to offload computational load of running model from raspberry pi to another machine
- Send frames from main client (Raspberry pi) to a server (Old linux laptop)
- Server runs object detection using selected model, results from server is then sent back to client
- client recieves 
"""
from detector import run_full_detection_pipeline, fcos_resnet50
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
