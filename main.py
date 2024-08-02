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
    







##############################################
# import torch
# import cv2
# import mediapipe as mp

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# # Function to detect humans in a frame
# def detect_humans(frame):
#     results = model(frame)
#     return results

# # Function to estimate pose
# def estimate_pose(frame):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#     return results

# # Read video frames
# cap = cv2.VideoCapture('path_to_your_video.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect humans
#     results = detect_humans(frame)
#     for *xyxy, conf, cls in results.xyxy[0]:
#         if int(cls) == 0:  # Class 0 is 'person'
#             x1, y1, x2, y2 = map(int, xyxy)
#             human_frame = frame[y1:y2, x1:x2]

#             # Estimate pose
#             pose_results = estimate_pose(human_frame)
#             if pose_results.pose_landmarks:
#                 mp.solutions.drawing_utils.draw_landmarks(human_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             frame[y1:y2, x1:x2] = human_frame

#     cv2.imshow('YOLO and MediaPipe', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


##############################################
    

########## Make BG diff with estimated background ###########
                # # Convert the background to grayscale
                # background_gray = cvtColor(background, COLOR_BGR2GRAY)

                # # Convert current frame to grayscale
                # gray = cvtColor(frame, COLOR_BGR2GRAY)

                # # Compute the absolute difference between the current frame and the previous frame
                # frame_diff = absdiff(background_gray, gray)

                # # Apply a binary threshold to the difference image
                # _, thresh = threshold(frame_diff, 80, 255, THRESH_BINARY)

                # kernel_open = np.ones((20, 20), np.uint8) # remove small noise
                # kernel_close = np.ones((50, 50), np.uint8) # fill gaps
                # thresh = morphologyEx(thresh, MORPH_CLOSE, kernel_close)
                # thresh = morphologyEx(thresh, MORPH_OPEN, kernel_open)

                # # Find contours in the thresholded image
                # contours, _ = findContours(thresh, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

                # # Draw bounding boxes around significant contours
                # for contour in contours:
                #     bbox = boundingRect(contour)
                #     frame = draw_box(frame, bbox)

                #################################
    

# Perform traditional human detection
                # bbox = template_matching(frame, template)

                # update the template for next iteration
                # template = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                
                ########## Make BG diff with consecutive frames ###########
                # # Convert current frame to grayscale
                # gray = cvtColor(frame, COLOR_BGR2GRAY)

                # # Init first frame
                # if previous_gray is None:
                #     previous_gray = gray
                #     continue

                # # Compute the absolute difference between the current frame and the previous frame
                # frame_diff = absdiff(previous_gray, gray)
                # previous_gray = gray

                # # Apply a binary threshold to the difference image
                # _, thresh = threshold(frame_diff, 50, 255, THRESH_BINARY)

                # # Use morphological operations to remove noise and fill gaps
                # kernel_open = np.ones((10, 10), np.uint8) # remove small noise
                # kernel_close = np.ones((90, 90), np.uint8) # fill gaps
                # thresh = morphologyEx(thresh, MORPH_CLOSE, kernel_close)
                # thresh = morphologyEx(thresh, MORPH_OPEN, kernel_open)

                # # Find contours in the thresholded image
                # contours, _ = findContours(thresh, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

                # # Draw bounding boxes around significant contours
                # for contour in contours:
                #     bbox = boundingRect(contour)
                #     frame = draw_box(frame, bbox)
                #####################