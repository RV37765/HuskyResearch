#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import os
import time

ORIGIN_X   = -97.695411
ORIGIN_Y   = -39.422331
RESOLUTION = 0.05
HEIGHT     = 645

robot_x = 0.0
robot_y = 0.0
got_pose = False

def amcl_callback(msg):
    global robot_x, robot_y, got_pose
    robot_x = msg.pose.pose.position.x
    robot_y = msg.pose.pose.position.y
    got_pose = True

def map_to_pixel(map_x, map_y):
    pixel_x = int((map_x - ORIGIN_X) / RESOLUTION)
    pixel_y = int(HEIGHT - ((map_y - ORIGIN_Y) / RESOLUTION))
    return pixel_x, pixel_y

def main():
    global got_pose
    print("Script started - connecting to ROS...")
    rospy.init_node('terminal_map_viewer')
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, amcl_callback)
    print("Subscribed to /amcl_pose - waiting for data...")

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        if got_pose:
            px, py = map_to_pixel(robot_x, robot_y)
            os.system('clear')
            print("========== ROBOT POSITION ==========")
            print(f"Map:   x={robot_x:.2f}  y={robot_y:.2f}")
            print(f"Pixel: x={px}  y={py}")
            print("====================================")
        else:
            print("Waiting for amcl_pose...")
        rate.sleep()

if __name__ == '__main__':
    main()