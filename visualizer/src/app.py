import cv2
import sys
import numpy as np

from constants import *
from camera import Camera
from canvas import Canvas

from modelprocess import ModelProcess

import util

class App:

    def __init__(self):
        # Create windows.
        cv2.namedWindow(CAM_WINDOW)
        cv2.namedWindow(CAP_WINDOW)
        cv2.namedWindow(RES_WINDOW)

        # Create a model process.
        self.model_path = '../data/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
        self.model_process = ModelProcess(self.model_path)

        # Generate the legend used for the model.
        self.legend = util.generate_legend(self.model_path)
        self.label_count = len(util.generate_labels(self.model_path)[0])

        # Open a camera.
        self.camera = Camera(CAM_ID)

        # Live segmentation state.
        self.live_segment = False
        self.live_segment_ready = True

        # Processing state.
        self.processing = False
        self.loss = None

        # Image buffers.
        self.camera_frame = None
        self.camera_segmented = None
        self.im_captured = None
        self.im_segmented = None
        self.im_processed = None

        # Canvas.
        self.canvas = None

        # Key events.
        self.key = 0

        # Initialize callbacks.
        self.init_callbacks()

        # Start the camera.
        self.camera.start()

    def __del__(self):
        pass

    # Run a tick of the application.
    def tick(self):
        # Read from the camera.
        self.camera_frame = self.camera.get()
        if self.live_segment:
            if self.live_segment_ready:
                self.live_segment_ready = False
                self.model_process.submit(COMMAND_SEGMENT,
                                          (EVENT_CAMERA_SEGMENT, self.camera_frame))
        else:
            self.camera_segmented = None

        # Display the current camera frame.
        camera_combined = self.camera_frame
        if self.camera_segmented is not None:
            camera_combined = self.camera_frame // 3 + self.camera_segmented
        util.display_image(CAM_WINDOW, camera_combined)

        # Capture a pressed key.
        self.key = cv2.waitKey(1) & 0xff

        # Toggle live segmenting if the live segmenting key is pressed.
        if self.key_pressed(KEY_LIVE):
            self.live_segment = not self.live_segment

        # Capture a frame if the capture key is pressed.
        if self.key_pressed(KEY_CAPTURE):
            self.capture(self.camera_frame)

        # Open a file if the open key is pressed.
        if self.key_pressed(KEY_OPEN):
            path = input('path> ')
            try:
                image = cv2.imread(path)
                image = util.convert(image)
                image = cv2.resize(image, (CAM_WIDTH, CAM_HEIGHT))
                self.capture(image)
                print('Image loaded')
            except:
                print('Invalid path')

        # Export the segmented image if the export key is pressed.
        if self.key_pressed(KEY_EXPORT):
            if self.canvas is not None:
                path = input('path> ')
                try:
                    cv2.imwrite(path, util.convert(self.canvas.get_combined()))
                    print('Image saved')
                except:
                    print('Invalid path')

        # Fill the canvas if the fill key is pressed.
        if self.key_pressed(KEY_FILL):
            if self.canvas is not None:
                self.canvas.fill()

        # Process the segment map if the process key is pressed.
        if self.key_pressed(KEY_PROCESS):
            if self.canvas is not None:
                self.process(self.canvas.get_map())

        # Save the result if the save key is pressed.
        if self.key_pressed(KEY_SAVE):
            if self.im_processed is not None:
                path = input('path> ')
                try:
                    cv2.imwrite(path, util.convert(self.im_processed))
                    print('Image saved')
                except:
                    print('Invalid path')

        # Segment the result if the test key is pressed.
        if self.key_pressed(KEY_TEST):
            if self.im_processed is not None:
                self.capture(self.im_processed)

        # Quit if the quit key is pressed.
        if self.key_pressed(KEY_QUIT):
            self.camera.stop()
            self.model_process.stop()
            return False

        # Tick the model process.
        self.model_process.tick()

        return True

    # Initialize callbacks.
    def init_callbacks(self):
        # Camera segment callback.
        self.model_process.subscribe(EVENT_CAMERA_SEGMENT, self.cb_camera_segment)

        # Segment result callback.
        self.model_process.subscribe(EVENT_SEGMENT, self.cb_segment)

        # Process result callback.
        self.model_process.subscribe(EVENT_RESULT, self.cb_result)

        # Accuracy callback.
        self.model_process.subscribe(EVENT_ACCURACY, self.cb_accuracy)

        # Mouse events on canvas.
        cv2.setMouseCallback(CAP_WINDOW, self.cb_canvas_mouse)

    # Capture a frame to process.
    def capture(self, frame):
        if self.processing:
            return
        self.processing = True

        # Discard the canvas.
        self.canvas = None

        # Display the captured frame.
        self.im_captured = frame
        util.display_image(CAP_WINDOW, self.im_captured)

        # Segment the captured frame.
        self.model_process.submit(COMMAND_SEGMENT, (EVENT_SEGMENT, frame))

    # Process a segment map.
    def process(self, segment_map):
        if self.processing:
            return
        self.processing = True

        # Create a progress bar.
        sys.stdout.write('[%s]' % (' ' * ITERATIONS))
        sys.stdout.flush()
        sys.stdout.write('\b' * (ITERATIONS + 1))

        # Generate an adversarial image.
        self.model_process.submit(COMMAND_PROCESS, (self.im_captured, segment_map))

    # Callback for segmented camera data.
    def cb_camera_segment(self, data):
        self.camera_segmented = util.label_to_color_image(data).astype(np.uint8)

        self.live_segment_ready = True

    # Callback for segmented capture data.
    def cb_segment(self, data):
        self.im_segmented = data

        # Create a new canvas.
        self.canvas = Canvas(self.im_captured, self.legend, self.im_segmented, self.label_count)

        # Show the frame.
        self.canvas.update()

        self.processing = False

    # Callback for process result.
    def cb_result(self, data):
        if data is None:
            sys.stdout.write(']\n')
            self.processing = False
            self.model_process.submit(COMMAND_ACCURACY,
                                      (self.canvas.get_map(), self.im_processed, self.loss))
            return

        _, self.im_processed, self.loss = data

        # Display the processed image.
        util.display_image(RES_WINDOW, self.im_processed)

        # Update the progress bar.
        sys.stdout.write('=')
        sys.stdout.flush()

    # Callback for accuracy.
    def cb_accuracy(self, data):
        print(data)

    # Callback for mouse events on the canvas.
    def cb_canvas_mouse(self, evt, x, y, flags, params):
        if self.canvas is not None:
            self.canvas.mouse(evt, x, y)

    # Determine if a key has been pressed.
    def key_pressed(self, key):
        return self.key == ord(key)
