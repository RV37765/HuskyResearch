#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan

pub = None

def callback(msg):
    fixed = LaserScan()
    fixed.header = msg.header
    fixed.angle_min = msg.angle_min
    fixed.angle_increment = msg.angle_increment
    fixed.range_min = msg.range_min
    fixed.range_max = msg.range_max
    fixed.ranges = msg.ranges
    fixed.intensities = msg.intensities
    # Fix angle_max so Karto's formula gives exactly len(ranges):
    # Karto: round((angle_max - angle_min) / angle_increment) + 1 = N
    # Setting angle_max = angle_min + (N-1)*angle_increment makes it exact.
    n = len(msg.ranges)
    fixed.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
    pub.publish(fixed)

rospy.init_node('scan_relay')
pub = rospy.Publisher('/scan_fixed', LaserScan, queue_size=1)
rospy.Subscriber('/scan', LaserScan, callback)
rospy.spin()
