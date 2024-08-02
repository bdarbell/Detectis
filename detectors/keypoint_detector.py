#!/usr/bin/env python

# 3rd party imports
from ultralytics import YOLO
import numpy as np
from numpy import typing as npt


class KeypointDetector:
    """
    
    """
    def __init__(self) -> None:
        """
        Initialize the KeypointDetector.
        """
        self.model = YOLO("yolov8n-pose.pt")

    def detect(self, frame:npt.NDArray, rois:tuple, p:int = 20) -> npt.NDArray:
        """
        Detect keypoints in the frame.

        Args:
            frame (numpy.ndarray): The frame to detect keypoints in
            rois (tuple): The regions of interest to detect keypoints in
            p (int): The padding around the regions of interest (default: 20)

        Returns:
            npt.NDArray: A list of keypoints for the detected objects
        """

        keypoints_list = []
        for roi in rois:
            x1, y1, w, h = roi
            human_frame = frame[y1-p:y1+h+p, x1-p:x1+w+p]
            results = self.model.predict(human_frame, classes=[0], verbose=False)

            for result in results[0].numpy():
                keypoints = result.keypoints.xy.flatten().reshape((-1, 2))
                
                # Filter out (0, 0) keypoints
                keypoints = np.array([keypoint for keypoint in keypoints if keypoint[0] != 0 or keypoint[1] != 0])
                
                # Shift keypoints to global coordinates
                keypoints = keypoints + [x1-p, y1-p]
                
                keypoints_list.extend(keypoints)

        return keypoints_list