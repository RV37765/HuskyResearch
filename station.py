#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import PoseWithCovarianceStamped

last_pose    = None
is_moving    = False
static_pub   = None

def amcl_callback(msg):
    global last_pose
    last_pose = msg

def odom_callback(msg):
    global is_moving
    # check if robot is actually moving
    linear  = msg.twist.twist.linear
    angular = msg.twist.twist.angular
    speed   = abs(linear.x) + abs(linear.y) + abs(angular.z)
    is_moving = speed > 0.01  # moving threshold

def main():
    rospy.init_node('transform_anchor')
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, amcl_callback)
    rospy.Subscriber('/odometry/filtered', Odometry, odom_callback)
    
    rate = rospy.Rate(10)
    locked_pose = None
    
    while not rospy.is_shutdown():
        if last_pose is not None:
            if not is_moving:
                # robot is stationary — lock the transform
                if locked_pose is None:
                    locked_pose = last_pose
                    rospy.loginfo("Robot stationary — locking transform")
            else:
                # robot is moving — allow amcl to update freely
                locked_pose = None
                rospy.loginfo("Robot moving — unlocking transform")
        
        rate.sleep()

if __name__ == '__main__':
    main()