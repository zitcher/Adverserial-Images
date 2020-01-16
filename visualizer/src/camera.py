from threading import Lock, Thread

import numpy as np
import cv2
import time

from constants import *

import util

class Camera(Thread):

    def __init__(self, camera_id):
        Thread.__init__(self)

        self.running = False
        self.lock = Lock()

        # Initialize the camera feed.
        self.feed = cv2.VideoCapture(camera_id)
        self.valid = self.feed.isOpened()
        self.empty = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        self.frame = self.empty

        # Get properties.
        if self.valid:
            self.feed.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.feed.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    def __del__(self):
        if self.valid:
            # Release the camera feed.
            self.feed.release()

    # Start the camera.
    def run(self):
        self.running = True
        while self.running:
            self.lock.acquire()
            self.frame = self.capture()
            self.lock.release()

            time.sleep(0.017)

    # Stop the camera.
    def stop(self):
        self.running = False

    # Get the current frame.
    def get(self):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        return frame

    # Capture a frame.
    def capture(self):
        if not self.valid:
            return self.empty

        # Read a frame.
        ret, frame = self.feed.read()

        # Resize the frame.
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        # Convert the frame to RGB.
        frame = util.convert(frame)

        if not ret:
            # Something went wrong with the capture.
            return self.empty

        return frame

    # Get the size of the image.
    def size(self):
        if not self.valid:
            return (0, 0)

        return (self.width, self.height)
