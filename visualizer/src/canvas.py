import cv2
import numpy as np

from constants import *

import util

class Canvas:

    def __init__(self, image, legend, segment_map, label_count):
        self.image = image
        self.legend = legend
        self.segment_map = segment_map.astype(np.uint8)
        self.label_count = label_count
        self.combined = image

        self.selected = None
        self.drawing = False

    def __del__(self):
        pass

    def get_map(self):
        return self.segment_map

    def get_combined(self):
        return self.combined

    def mouse(self, evt, x, y):
        if evt == cv2.EVENT_LBUTTONDOWN:
            # Mouse down.
            if x < CAP_WIDTH:
                # Start drawing.
                self.drawing = True
                if self.selected:
                    self.paint(x, y)
            else:
                # Legend selection.
                for i in range(self.label_count):
                    box_x = CAP_WIDTH + PALETTE_SPACING
                    box_y = PALETTE_SPACING + (PALETTE_SIZE + PALETTE_SPACING) * i
                    box_w = PALETTE_SIZE
                    box_h = PALETTE_SIZE
                    if x >= box_x and y >= box_y and x < box_x+box_w and y < box_y+box_h:
                        self.selected = i
                        break

        if evt == cv2.EVENT_LBUTTONUP:
            # Stop drawing.
            self.drawing = False

        if evt == cv2.EVENT_MOUSEMOVE and self.drawing and self.selected is not None:
            # Paint on the canvas.
            self.paint(x, y)

        # Update the canvas.
        self.update()

    def fill(self):
        if self.selected is not None:
            self.segment_map = np.ones(self.segment_map.shape, dtype=np.uint8) * self.selected

    def paint(self, x, y):
        cv2.circle(self.segment_map, (x, y), 20, self.selected, -1)

    def update(self):
        # Overlay the segmented frame.
        segmented_image = util.label_to_color_image(self.segment_map).astype(np.uint8)
        self.combined = self.image // 6 + segmented_image

        # Resize the frame.
        self.combined = cv2.resize(self.combined, (CAP_WIDTH, CAP_HEIGHT))

        # Update the legend.
        legend = self.legend.copy()
        if self.selected is not None:
            x = PALETTE_SPACING
            y = PALETTE_SPACING + (PALETTE_SIZE + PALETTE_SPACING) * self.selected
            w = PALETTE_SIZE
            h = PALETTE_SIZE
            cv2.rectangle(legend, (x,y), (x+w,y+h), (255, 255, 255), 2)
            cv2.rectangle(legend, (x,y), (x+w,y+h), (0, 0, 0), 1)

        # Show the frame.
        util.display_image(CAP_WINDOW, np.concatenate((self.combined, legend), axis=1))
