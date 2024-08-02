#!/usr/bin/env python

# 3rd party imports
from cv2 import cvtColor, COLOR_BGR2GRAY, absdiff, threshold, THRESH_BINARY, morphologyEx, MORPH_CLOSE, MORPH_OPEN, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, boundingRect
import numpy as np
from numpy import typing as npt


class TraditionalObjectDetector:
    """
    Detect objects using consecutive frames of a video.
    This class uses background subtraction to detect objects.
    """
    def __init__(self) -> None:
        """
        Initialize the ObjectDetector.
        """
        self.previous_gray = None

    def detect(self, frame:npt.NDArray) -> list:
        """
        Detect objects in the frame.

        Args:
            frame (numpy.ndarray): The frame to detect objects in

        Returns:
            list: A list of bounding boxes for the detected objects
        """
        # Convert current frame to grayscale
        gray = cvtColor(frame, COLOR_BGR2GRAY)

        # Init first frame
        if self.previous_gray is None:
            self.previous_gray = gray
            return []

        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = absdiff(self.previous_gray, gray)
        self.previous_gray = gray

        # Apply a binary threshold to the difference image
        _, thresh = threshold(frame_diff, 50, 255, THRESH_BINARY)

        # Use morphological operations to remove noise and fill gaps
        kernel_open = np.ones((20, 20), np.uint8)
        kernel_close = np.ones((90, 90), np.uint8)
        thresh = morphologyEx(thresh, MORPH_CLOSE, kernel_close) # remove small noise
        thresh = morphologyEx(thresh, MORPH_OPEN, kernel_open) # fill gaps

        # Find contours in the thresholded image
        contours, _ = findContours(thresh, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around significant contours
        bounding_boxes = [boundingRect(contour) for contour in contours]

        return bounding_boxes