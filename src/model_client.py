import cv2
import time
import requests
import numpy as np


def encoded_img_to_bytes(img: np.ndarray) -> bytes:
    # encode image
    _, encoded_img = cv2.imencode(".png", img)

    # convert to numpy array
    np_encoded_img = np.array(encoded_img)

    # convert the array to bytes
    encoded_bytes = np_encoded_img.tobytes()
    return encoded_bytes


def run_app(server_url):

    CAM = cv2.VideoCapture(0)
    while CAM.isOpened():

        ret, frame = CAM.read()

        if not ret:
            print("Unable to read frame from camera..")
            break

        # Prepare the POST request to send the image
        try:
            encoded_bytes = encoded_img_to_bytes(frame)
            start_time = time.time()
            response = requests.post(server_url, encoded_bytes)
            end_time = time.time()
            if response.status_code == 200:
                results = response.json()
                results["metrics"]["server_latency"] = end_time - start_time
                print(results)
            else:
                print(f"Error: Server responded with status code {response.status_code}")
        except Exception as e:
            print(f"Error sending image to server: {e}")
            break


server_url = "127.0.0.1"
run_app(f"http://{server_url}:8000/images")


