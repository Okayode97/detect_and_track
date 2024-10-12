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


def decode_bytes_to_img(bytes: bytes) -> np.ndarray:
    img = np.asarray(bytearray(bytes), dtype="uint8")
    decoded_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return decoded_img


def stream_camerafeed_to_server(server_url):

    CAM = cv2.VideoCapture(1)
    # camera broken on own laptop :(
    while CAM.isOpened():

        ret, frame = CAM.read()

        if not ret:
            print("Unable to read frame from camera..")
            break

        # Prepare the POST request to send the image
        try:
            encoded_bytes = encoded_img_to_bytes(frame)
            response = requests.post(server_url, encoded_bytes)
            if response.status_code == 200:
                print(f"Frame sent successfully | Returned content: {response.content}")
            else:
                print(f"Error: Server responded with status code {response.status_code}")
        except Exception as e:
            print(f"Error sending image to server: {e}")
            break

        # You can add a sleep time if you want to slow down the frame rate
        time.sleep(0.1)  # 100ms delay between frames


stream_camerafeed_to_server("http://127.0.0.1:8000/images")


