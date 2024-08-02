#!/usr/bin/env python

# 3rd party imports
from ultralytics import YOLO
from numpy import typing as npt

class DeepLearningObjectDetector:
    """
    Detect objects in consecutive frames.
    This class uses deep learning to detect objects.
    """
    def __init__(self, humans_only:bool = True) -> None:
        """
        Initialize the DeepLearningObjectDetector.

        Args:
            humans_only (bool): Whether to detect only humans (default: True)
        """
        self.model = YOLO("yolov8n.pt")
        self.humans_only = humans_only

    def detect(self, frame:npt.NDArray) -> list:
        """
        Detect objects in the frame.

        Args:
            frame (numpy.ndarray): The frame to detect objects in

        Returns:
            list: A list of bounding boxes for the detected objects
        """
        # Perform object detection
        results = self.model.predict(frame, classes=[0] if self.humans_only else None, verbose=False)

        # Extract bounding boxes
        bboxes = [bbox for bbox in results[0].numpy().boxes.xywh]

        # Shift bounding boxes to top-left corner and convert to integers
        bboxes = [(int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2), int(bbox[2]), int(bbox[3])) for bbox in bboxes]

        return bboxes