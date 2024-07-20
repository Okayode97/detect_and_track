import cv2
import pytest
import numpy as np
from tracker import KF_filter, KalmanFilter, Q_discrete_white_noise, get_bbox_centre, Tracker, Track


class Demo_tracker:

    def simple_linear_diagonal_case(self):
        img = np.zeros((600, 600, 3))
        red = (0, 0, 255)
        green = (0, 255, 0)
        thickness = 2

        # box height & width
        box_hw = 10

        # intialise filter and read first detection
        box_filter = KF_filter(np.array([0, 0, box_hw, box_hw]), dt=1)
        
        for i in range(1, 600, 1):
            start_point = np.array([i, i])

            # update the filter
            box_filter.update(np.append(start_point, np.array([box_hw, box_hw]), axis=0))

            # get the estimated position and velocity
            estimated_position, _ = box_filter.get_estimated_state()

            image = img.copy()
            
            # draw the initial detection
            end_point = start_point + 10
            image = cv2.rectangle(image, start_point, end_point, red, thickness)

            # draw the estimated state
            estimated_position_start_point = estimated_position[:2].astype(int)
            estimated_position_end_point = estimated_position_start_point + estimated_position[-2:].astype(int)
            image = cv2.rectangle(image, estimated_position_start_point, estimated_position_end_point,
                                    green, thickness) 

            cv2.imshow("Blank image", image)
            if cv2.waitKey(1) == ord('q'):
                break

    def filter_on_tracking_mouse_cursor(self):
        # Initialize cursor position
        cursor_position = (0, 0)

        # Function to update the cursor position
        def update_cursor_position(event, x, y, flags, param):
            nonlocal cursor_position
            if event == cv2.EVENT_MOUSEMOVE:
                cursor_position = (x, y)

        # Initialize the blank screen
        width, height = 640, 480
        blank_screen = np.zeros((height, width, 3), dtype=np.uint8)

        box_hw = np.array([10, 10])

        measurement_color = (0, 255, 0)
        estimated_color = (255, 0, 0)

        # intiialise filter and read first position
        box_filter = KF_filter(detection=np.append(cursor_position, box_hw, axis=0), dt=1)

        # Set up the window and mouse callback
        cv2.namedWindow('Cursor Tracker')
        cv2.setMouseCallback('Cursor Tracker', update_cursor_position)

        while True:
            box_filter.update(np.append(cursor_position, box_hw, axis=0))

            estimated_position, estimated_velocity = box_filter.get_estimated_state()

            # Copy the blank screen
            frame = blank_screen.copy()
            
            # Draw the cursor position
            cursor_endpoint = cursor_position + box_hw
            cv2.rectangle(frame, cursor_position, cursor_endpoint, measurement_color, 1)

            # Draw the estimated position
            estimated_startpoint = estimated_position[:2].astype(int)
            estimated_endpoint = estimated_startpoint + estimated_position[-2:].astype(int)
            cv2.rectangle(frame, estimated_startpoint, estimated_endpoint, estimated_color, 1) 
            
            # Display the frame
            cv2.imshow('Cursor Tracker', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cv2.destroyAllWindows()

    def simple_tracker(self):
        # track position and velocity using a sensor that reads only position
        f = KalmanFilter(dim_x=2, dim_z=1)

        # set the initial state variable
        f.x = np.array([0., 0.])

        # define initial state covariance matrix
        f.P *= 1000

        # define state transition variable
        f.F = np.array([[1., 1.],
                        [0., 1.]])

        # define measurement function
        f.H = np.array([[1., 0.]])

        # define measurement covariance matrix
        f.R = np.array([[2.]])

        # define process noise 
        f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)


        for i in range(1, 20, 1):
            f.predict()
            f.update(i)

            print(f"measurement: {i} | prediction: {f.x}")

    
class TestKF_filter:
    
    @pytest.fixture
    def diagonal_bbox(self):
        bbox_detections = []

        for i in range(0, 60, 1):
            bbox_detections.append(np.array([i, i, 10, 10]))
        return bbox_detections


    def test_simple_linear_motion(self, diagonal_bbox):
        box_filter = KF_filter(detection=diagonal_bbox[0], dt=1)

        for i, bbox_detection in enumerate(diagonal_bbox[1:]):
            # test performance by dropping different detections
            if i not in [15, 30, 45, 60]:
                box_filter.update(bbox_detection)
            else:
                box_filter.update(np.array([]))
            position, velocity = box_filter.get_estimated_state()

            np.testing.assert_allclose(bbox_detection, position, atol=0.1)

            # give the model enough time to get a better understanding of motion of the box
            if i > 10:
                np.testing.assert_allclose(np.array([1, 1, 0, 0]), velocity, atol=0.1)

@pytest.mark.parametrize("bbox, expected_centre", [(np.array([0, 0, 10, 10]), np.array([5, 5])),
                                                    (np.array([34, 52, 120, 80]), np.array([ 74., 112.])),
                                                    (np.array([210, 178, 95, 130]), np.array([275. , 225.5])),
                                                    (np.array([89, 44, 150, 110]), np.array([144., 119.])),
                                                    (np.array([300, 250, 75, 200]), np.array([400. , 287.5])),
                                                    ])
def test_get_bbox_centre(bbox, expected_centre):
    np.testing.assert_array_equal(get_bbox_centre(bbox), expected_centre)


class TestTracker:

    @pytest.fixture()
    def instantiated_tracker(self):
        kf_filter_tracker = Tracker()

        # initialise first detection
        bbox_horizontal = np.array([2, 0, 10, 10]) # changing y
        bbox_vertical = np.array([0, 0, 10, 10]) # changing x

        # initialise filters
        KF_filter_horizontal = KF_filter(bbox_horizontal, 1.0)
        KF_filter_vertical = KF_filter(bbox_vertical, 1.0)

        # initialise tracks
        kf_filter_tracker.list_of_tracks.append(Track(KF_filter_horizontal, 1))
        kf_filter_tracker.list_of_tracks.append(Track(KF_filter_vertical, 2))
        
        # update filters for 10 iterations
        # done to allow filter to develop a model of the detection's movements
        for i in range(1, 10, 1):            
            kf_filter_tracker.list_of_tracks[0].filter.update(np.array([2, i, 10, 10]))
            kf_filter_tracker.list_of_tracks[1].filter.update(np.array([i, 0, 10, 10]))

        
        # check that filter correctly estimates next state of detection after 10 iterations
        # track 0, horizontal. track 1, vertical.
        np.testing.assert_allclose(kf_filter_tracker.list_of_tracks[0].filter.get_estimated_state()[0],
                                      np.array([2, 10, 10, 10]), rtol=0.1)
        np.testing.assert_allclose(kf_filter_tracker.list_of_tracks[1].filter.get_estimated_state()[0],
                                      np.array([10, 0, 10, 10]), rtol=0.1)
        
        return kf_filter_tracker
        

    def test_associate_detections_to_tracks_correctly_associates_detections(self, instantiated_tracker):
        # detections placed in order of horizontal, vertical
        detections = np.array([[2, 11, 10, 10], [11, 0, 10, 10]])

        # instantiated tracker placed in order of horizontal, vertical
        association = instantiated_tracker.associate_detections_to_tracks(detections)
        
        assert association[0] == 0
        assert association[1] == 1

        # ------next time step----------

        # detections placed in order of vertical, horizontal
        detections = np.array([[12, 0, 10, 10], [2, 12, 10, 10]])

        # instantiated tracker placed in order of horizontal, vertical
        association = instantiated_tracker.associate_detections_to_tracks(detections)
        
        assert association[0] == 1
        assert association[1] == 0
    
    def test_associate_detections_function_does_not_create_associations_for_new_detections(self, instantiated_tracker):
        # detections placed in order of horizontal, vertical, new detection
        detections = np.array([[2, 11, 10, 10], [11, 0, 10, 10], [15, 15, 10, 10]])

        # instantiated tracker placed in order of horizontal, vertical
        association = instantiated_tracker.associate_detections_to_tracks(detections)
        
        assert association[0] == 0
        assert association[1] == 1

if __name__ == "__main__":
    Demo_tracker().filter_on_tracking_mouse_cursor()