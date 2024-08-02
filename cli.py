#!/usr/bin/env python

"""
Command line interface.
"""

# local imports
import os

# standard imports
import argparse

# local imports
from .main import start_program

# Current file path
DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "out")

def cli() -> None:
    """
    Command line interface.
    """
    parser = argparse.ArgumentParser()

    # Define options
    parser.add_argument("path", help="Input path to the video file")
    parser.add_argument("-m", "--method", choices=["traditional", "deeplearning"], default="deeplearning", help="Method to use for object detection (default: %(default)s)")
    parser.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR, help="Output directory (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("-k", "--detect_keypoints", action="store_true", help="Detect keypoints")


    # Parse options
    args = parser.parse_args()

    # Run the program
    run(args)


def run(args: argparse.Namespace) -> None:
    """
    Runs the program with the selected options.

    Args:
        args (argparse.Namespace): The command line arguments
    """

    # Run program
    start_program(args.path, args.method, args.outdir, detect_keypoints=args.detect_keypoints, verbose=args.verbose)
