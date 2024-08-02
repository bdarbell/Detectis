#!/usr/bin/env python

# 3rd party imports
from cv2 import circle, rectangle
from numpy import typing as npt

def draw_boxes(image:npt.NDArray, boxes:tuple, color:tuple = (0, 0, 255), thickness:int = 2) -> npt.NDArray:
    """
    Draw a box on the image.

    Args:
        image (numpy.ndarray): The image to draw on
        boxes (tuple): The boxes to draw (x, y, width, height)
        color (tuple): The color of the box (default: (0, 0, 255) - red)
        thickness (int): The thickness of the box (default: 2)

    Returns:
        numpy.ndarray: The image with the box drawn
    """
    # Draw boxes
    for box in boxes:
        x, y, w, h = box
        image = rectangle(image, (x, y), (x + w, y + h), color, thickness)

    return image


def draw_points(image:npt.NDArray, points:npt.NDArray, point_color: tuple = (255, 0, 0)) -> npt.NDArray:
    """
    Draw points on the image.

    Args:
        image (numpy.ndarray): The image to draw on
        points (numpy.ndarray): The points to draw
        point_color (tuple): The color of the points (default: (255, 0, 0) - blue)

    Returns:
        numpy.ndarray: The image with the points drawn
    """
    # Draw points
    for p in points:
        point = tuple(p.astype(int)) 
        circle(image, point, 3, point_color, -1)
    
    return image