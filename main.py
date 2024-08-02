#!/usr/bin/env python

# standard imports
import os

# local imports
from .detectors import DeepLearningObjectDetector, KeypointDetector, TraditionalObjectDetector
from .utils import draw_boxes, draw_points
from .utils import VideoGenerator, VideoReader

# 3rd party imports
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

COURT_BOUNDARIES = ((30, 670), (1250, 670), (300, 170), (980, 170))


def start_program(video_path:str, method:str, outdir:str, detect_keypoints:bool = False, verbose:bool = False) -> None:
    """
    Start the program.

    Args:
        video_path (str): The path to the video file
        method (str): The method to use for object detection
        outdir (str): The output directory
        verbose (bool): Whether to print verbose output
    """
    if verbose:
        print(f"Starting program with video: {video_path}, method: {method} and output directory: {outdir}")

    # Initialize the video utils
    video_reader = VideoReader(video_path)
    video_generator = VideoGenerator(os.path.join(outdir, f"output_{method}_keypoints.mp4" if detect_keypoints else f"output_{method}.mp4"), video_reader.fps)

    # Initialize detectors
    object_detector = TraditionalObjectDetector() if method == "traditional" else DeepLearningObjectDetector()

    if detect_keypoints:
        keypoints_detector = KeypointDetector()

    # Define court
    court_polygon = Polygon(COURT_BOUNDARIES)

    # Loop through the frames of the video        
    for frame in tqdm(video_reader.frames(), desc="Processing video", total=video_reader.length):
        if frame is None:
            print("Error: Image not loaded properly")

        # Perform object detection
        try:
            bboxes = object_detector.detect(frame)

        except Exception as e:
            print(f"Detection failed. Error: {e}")
            break

        # Filter out bounding boxes that are not in the court
        bboxes = [bbox for bbox in bboxes if court_polygon.contains(Point(bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))]

        # Perform body keypoints prediction in the bounding boxes
        if detect_keypoints:
            try:
                keypoints = keypoints_detector.detect(frame, bboxes)

            except Exception as e:
                print(f"Keypoint detection failed. Error: {e}")
                break

        # Draw detections
        frame = draw_boxes(frame, bboxes)
        if detect_keypoints:
            frame = draw_points(frame, keypoints)

        # Append the frame to the video
        video_generator.append(frame)
    
    # Release the video reader
    video_reader.release()

    # Generate the output video
    video_generator.generate()
    