# Detectis

## Description

This project focuses on the detection of athletes "in action" during a tennis match, employing two distinct approaches: a classical computer vision method and a deep learning-based method. Additionally, the detected athletes are subjected to pose estimation using a separate deep learning model.

#### Key Features:
##### Classical Computer Vision Approach:
- Utilizes traditional pixel-based analysis.
- Processes video frames to detect athletes by analyzing differences between consecutive frames.
- Efficient and suitable for real-time applications.

##### Deep Learning Approach:
- Employs the YOLOv8 nano model for robust human detection.
- Capable of identifying athletes, referees, and spectators, with additional filtering to focus on athletes.
- Provides stable and accurate detections, albeit at a lower processing speed.

##### Additional Analysis:
- Post-detection, a separate deep learning model performs keypoint estimation on the detected athletes. This allows for detailed analysis of their movements and positions during the match, enhancing the overall understanding of their performance.


## Installation

To run this project, the dependencies mentioned in the file [requirements.txt](requirements.txt) are required.

They can be installed with the following command:
```bash
pip install requirements.txt
```

## Usage

```bash
python -m Detectis <path> [-m <method>] [-o <outdir>] [-v] [-k]
```
Replace <path> with the path to your video file.

>**NOTE**: The commands must be run from outside the project folder. For example:
>- Path to project: `.../path/to/project/Detectis`
>- Run the command from `.../path/to/project`

### Options
- `-m` or `--method`: Choose the method to use for object detection. The options are `traditional` or `deeplearning`. The default is `deeplearning`.
- `-o` or `--outdir`: Specify the output directory. The default is `Detectis/out`.
- `-v` or `--verbose`: Enable verbose output. This option does not take any value.
- `-k` or `--detect_keypoints`: Enable keypoint detection. This option does not take any value.

### Example
- Process video with the default options:
```bash
python -m Detectis Detectis/media/Wimbledon-2023.mp4
```

- Process video using the traditional method and enable verbose output:
```bash
python -m Detectis Detectis/media/Wimbledon-2023.mp4 -m traditional -v
```

- Process video and save the output to a specific directory:
```bash
python -m Detectis Detectis/media/Wimbledon-2023.mp4 -o /path/to/output
```

For more information on the options, use the `-h` or `--help` flag:
```bash
python -m Detectis --help
```