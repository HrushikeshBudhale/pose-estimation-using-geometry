#!/usr/bin/env python

import rospy
from bar_detector.detect_bars import BarDetector

if __name__ == '__main__':
    rospy.init_node('BarDetector', anonymous=False)
    bar_detector_obj = BarDetector()
    rospy.spin()