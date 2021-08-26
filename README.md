# Pose Estimation from Horizontal and Vertical Bars

This project contains a ROS package required to localized a robot position in a structured environment using a monocular camera. This package receives a binary images after inferencing from a DL model which segments vertical and horizontal bars visible to the camera frame. After receiving these images the ROS node processes images by applying image filters and geometric transforms to estimate the current position of the camera. Extimated pose is then publisehd on a respective topics.
