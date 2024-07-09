import cv2
import numpy as np
from tracker import KF_filter, Detection, Tracker


# def test_simple_linear_diagonal_case():
#     box_filter = Filter(dt=1/9)

#     # diagonal movement
#     img = np.zeros((600, 600, 3))
#     color = (0, 0, 255)
#     thickness = 2

#     start_point = np.array([0, 0])
#     height = 10
#     width = 10

#     box_filter.update(np.array([0, 0, 10, 10]))

#     for i in range(0, 600, 1):
#         start_point = np.array([i, i])

#         # update the filter
#         print(start_point, np.array([10, 10]))
#         # box_filter.update(np.append(start_point, np.array([10, 10]), axis=0))

#         # # get the filter detection
#         # estimated_position, _ = box_filter.get_estimated_state()

#         image = img.copy()
#         image = cv2.rectangle(image, start_point, start_point, color, thickness) 
#         # image = cv2.rectangle(image, (int(estimated_position[0]), int(estimated_position[1])),
#         #                             (int(estimated_position[0]+estimated_position[2]),int(estimated_position[1]+estimated_position[3])),
#         #                     (0, 255, 0), thickness) 

#         cv2.imshow("Blank image", image)
#         if cv2.waitKey(1) == ord('q'):
#             break

# def test_filter_on_tracking_mouse_cursor():
#     # Function to update the cursor position
#     def update_cursor_position(event, x, y, flags, param):
#         global cursor_position
#         if event == cv2.EVENT_MOUSEMOVE:
#             cursor_position = (x, y)

#     # Initialize the blank screen
#     width, height = 640, 480
#     blank_screen = np.zeros((height, width, 3), dtype=np.uint8)

#     # Set up the window and mouse callback
#     cv2.namedWindow('Cursor Tracker')
#     cv2.setMouseCallback('Cursor Tracker', update_cursor_position)

#     # Initialize cursor position
#     cursor_position = (0, 0)

#     box_filter = Filter(dt=1/100)
#     box_filter.update(np.append(cursor_position, np.array([10, 10]), axis=0))

#     while True:

#         estimated_position, _ = box_filter.get_estimated_state()

#         # Copy the blank screen
#         frame = blank_screen.copy()
        
#         # Draw the cursor position
#         cv2.rectangle(frame, cursor_position, (cursor_position[0]+10, cursor_position[1]+10), (0, 255, 0), 1)

#         cv2.rectangle(frame, (int(estimated_position[0]), int(estimated_position[1])),
#                             (int(estimated_position[0]+estimated_position[2]), int(estimated_position[1]+estimated_position[3])),
#                                 (0, 255, 0), 1) 
        
#         # Display the frame
#         cv2.imshow('Cursor Tracker', frame)
        
#         box_filter.update(np.append(cursor_position, np.array([10, 10]), axis=0))

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Clean up
#     cv2.destroyAllWindows()


class TestDetection:

    def test_detection_are_properly_populated(self):
        simple_bounding_box = Detection(np.array([0, 0, 10, 10]))
        np.testing.assert_array_equal(simple_bounding_box.bbox, np.array([0, 0, 10, 10]))
    
    def test_detections_properly_convert_xyxy_to_xyhw(self):
        simple_bounding_box = Detection()
        simple_bounding_box.convert_xyxy_to_xyhw(np.array([10, 10, 30, 20]))
        np.testing.assert_array_equal(simple_bounding_box.bbox, np.array([10, 10, 20, 10]))
    
    def test_detections_can_be_conveted_back_to_xyxy(self):
        simple_bounding_box = Detection(np.array([10, 10, 20, 10]))
        xyxy = simple_bounding_box.convert_xyhw_to_xyxy()
        np.testing.assert_array_equal(xyxy, np.array([10, 10, 30, 20]))
    
    