import numpy as np
from tracker import Tracker
import cv2

# Parameters
image_width, image_height = 640, 480
num_boxes = 5
box_size = 50
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
max_steps = 500  # Number of steps in the animation
step_delay = 50  # Delay between steps in milliseconds

# Initialize box positions and velocities
np.random.seed(0)
positions = np.random.rand(num_boxes, 2) * [image_width - box_size, image_height - box_size]
velocities = (np.random.rand(num_boxes, 2) - 0.5) * 10  # Random velocities


# Function to update the positions of the bounding boxes
def update_positions() -> np.ndarray:
    bbox_detections: np.ndarray = np.empty((0, 4))

    for i in range(num_boxes):
        positions[i] += velocities[i]

        # Check for collision with walls and reverse velocity if necessary
        if positions[i][0] < 0 or positions[i][0] > image_width - box_size:
            velocities[i][0] = -velocities[i][0]
        if positions[i][1] < 0 or positions[i][1] > image_height - box_size:
            velocities[i][1] = -velocities[i][1]

        # Save current bounding box data
        x, y = positions[i]
        w, h = box_size, box_size
        bbox_detections = np.vstack([bbox_detections, np.array([int(x), float(y), float(w), float(h)])])

    return bbox_detections

# Create a blank image
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

bbox_tracker = Tracker()
# Create animation
for step in range(max_steps):
    # Clear the image
    image.fill(0)

    # Update positions
    bbox_detections = update_positions()
    bbox_tracker.update_filters(bbox_detections)

    # Draw bounding boxes
    for bbox_detection_, color in zip(bbox_detections, colors):
        x, y, h, w = bbox_detection_
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(image, top_left, bottom_right, color, 2)

    # Display the image
    cv2.imshow('Animation', image)
    if cv2.waitKey(step_delay) & 0xFF == 27:  # Press 'Esc' to exit
        break


cv2.destroyAllWindows()
