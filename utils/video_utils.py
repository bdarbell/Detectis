#!/usr/bin/env python

# standard imports
import os

# 3rd party imports
from cv2 import CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, VideoCapture, VideoWriter, VideoWriter_fourcc
from numpy import typing as npt


class VideoGenerator:
    """
    A class to generate a video from frames.
    """
    def __init__(self, output_path:str, fps:int = 30) -> None:
        """
        Initialize the VideoGenerator.

        Args:
            output_path (str): The path to save the video
            fps (int): The frames per second of the video
        """
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        self._fourcc = VideoWriter_fourcc(*'mp4v')

    def append(self, frame:npt.NDArray) -> None:
        """
        Append a frame to the video.

        Args:
            frame (numpy.ndarray): The frame to append
        """
        self.frames.append(frame)

    def generate(self) -> None:
        """
        Generate the video from the appended frames.
        """
        # Make sure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Get the height and width of the frames
        height, width, _ = self.frames[0].shape

        # Create the video writer
        video_writer = VideoWriter(self.output_path, self._fourcc, self.fps, (width, height))

        # Write each frame to the video
        for frame in self.frames:
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        # Inform the user
        print(f"Video saved to: {self.output_path}")


class VideoReader:
    """
    A class to read a video and yield frames.
    """
    def __init__(self, video_path:str) -> None:
        """
        Initialize the VideoReader.

        Args:
            video_path (str): The path to the video file
        """
        self.video_path = video_path
        self.cap = VideoCapture(self.video_path)
        self.fps = self.cap.get(CAP_PROP_FPS)
        self.length = self.cap.get(CAP_PROP_FRAME_COUNT)
    
    def release(self) -> None:
        """
        Release the video capture.
        """
        # Release the video
        self.cap.release()

    def frames(self) -> npt.NDArray:
        """
        Read the frames from the video.

        Yield:
            numpy.ndarray: The frames of the video
        """
        # Loop through the video
        while self.cap.isOpened():
            # Read the next frame
            _, frame = self.cap.read()

            # If the frame is None, then we have reached the end of the video
            if frame is None:
                break

            # Yield the frame
            yield frame