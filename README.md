# Pose Estimation in Structured Environment

### Author
- Hrushikesh Budhale

## Table of Contents
   * [What is this?](#what-is-this)
   * [How it works?](#how-it-works)
   * [Input](#input)
   * [Output](#output)
   * [Requirements](#requirements)
   * [Installation instructions](#installation-instructions)
   * [Execution](#execution)

## What is this?

This project contains a ROS package written for pose estimation in a structured environment using visual odometry from single monocular camera and knowledge about the surrounding environment. This package makes use of projective geometry and predictive filtering for achieving centimeter level accuracy in pose estimation.

## How it works?
1. This package receives 2 gray scale images after inference from a DL model which segments vertical and horizontal beams visible in the camera frame.

2. After receiving these images this ROS node processes images by applying temporal filters along with knowledge based filters and geometric transforms to estimate the current position of the camera.

3. Estimated pose is then utilized iteratively to predict next pose based on state variables.

4. Predicted pose is then updated from the observations and it gets published on respective ros topics.

## Input
- Input1: Inference from Horizontal beam detector model.

- Input2: Inference from Vertical beam detector model.

## Output
- This package generates odometry output containing X, Y, Z and Yaw of the robot.
- This package also provides estimate of Aisle width (one of the known structural value) which can be used as a metric of confidence at that timestamp.
- For debugging purpose, this package provides image overlay topic. (This requires original RGB image with timestamp as input)

<img src="https://github.com/HrushikeshBudhale/pose-estimation-using-geometry/blob/main/docs/pose_estimate.gif?raw=true" width="640" alt="Open loop pic">

## Requirements

- Ubuntu 18.04 (or above)
- ROS Melodic (or above)
- Python 3.6
- Numpy
- Opencv3

## Installation instructions

```
cd <catkin workspace>/src
sudo apt-get install git
git clone https://github.com/HrushikeshBudhale/pose-estimation-using-geometry
cd ..
catkin build
source ./devel/setup.bash
```

## Execution
This package depends on 3 input publishers publishing following data with time stamp,
- Rgb image
- gray-scale inference image of vertical beams
- gray-scale inference image of horizontal beams

If above data is available, launch the pose estimator node using following launch command.
```
roslaunch bar_detector bar_detector.launch
```
