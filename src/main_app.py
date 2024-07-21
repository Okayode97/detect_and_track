import cv2
import socket
from imagezmq import ImageSender
import logging

logger = logging.getLogger(__name__)


logging.basicConfig(filename='detect_and_tracker.log', level=logging.INFO)
CAM  = cv2.VideoCapture(0)
display_hostname = "127.0.0.1"

rpi_name = socket.gethostname() # send RPi hostname with each image
sender = ImageSender(connect_to=f'tcp://{display_hostname}:5555')

# instantiate detector

# instantiate tracker


while CAM.isOpened():
    ret, frame = CAM.read()
    if not ret:
        print("Unable to read frames from camera....")
        break

    print(f"frame being sent....")

    # get detection from frame

    # update tracker with detections

    # get tracker with prediction

    sender.send_image(rpi_name, frame)

    