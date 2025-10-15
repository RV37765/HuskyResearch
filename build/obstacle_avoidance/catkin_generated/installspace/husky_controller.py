#!/usr/bin/env python3
import rospy
import subprocess
import time
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class HuskyController:
    def __init__(self):
        # Set use_sim_time before initializing the node
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('husky_controller')
        
        # Wait for time to be properly set
        while rospy.Time.now().to_sec() == 0:
            rospy.loginfo("Waiting for /clock time to be set...")
            rospy.sleep(0.1)
        
        rospy.loginfo("Time is now set - starting controller")
        
        self.pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10)


        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        # Parameters
        self.min_distance = 1.2
        self.linear_speed = 0.4
        self.angular_speed = 0.5
        
        # State
        self.obstacle_detected = False
        self.current_linear = 0.0
        self.current_angular = 0.0
        
        # Initialize movement
        self.start_moving()

    def start_moving(self):
        """Initial movement to verify robot is working"""
        rospy.loginfo("Starting initial movement...")
        vel_msg = Twist()
        vel_msg.linear.x = 0.3  # Start with slower speed
        self.pub.publish(vel_msg)
        rospy.sleep(1)  # Move for 1 second
        rospy.loginfo("Initial movement complete")

    def laser_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[~np.isinf(ranges) & ~np.isnan(ranges)]
        
        if len(valid_ranges) == 0:
            rospy.logwarn("No valid laser measurements")
            self.stop()
            return
            
        min_distance = np.min(valid_ranges)
        min_angle_idx = np.argmin(msg.ranges)
        angle = msg.angle_min + min_angle_idx * msg.angle_increment
        
        rospy.loginfo(f"Min distance: {min_distance:.2f}, Angle: {angle:.2f}")
        
        if min_distance < self.min_distance:
            self.current_linear = 0.1
            turn_direction = 1 if angle > 0 else -1
            self.current_angular = turn_direction * self.angular_speed
            rospy.loginfo("Obstacle detected - turning")
        else:
            self.current_linear = self.linear_speed
            self.current_angular = 0.0
            rospy.loginfo("Path clear - moving forward")
        
        self.publish_velocity()

    def publish_velocity(self):
        vel_msg = Twist()
        vel_msg.linear.x = self.current_linear
        vel_msg.angular.z = self.current_angular
        self.pub.publish(vel_msg)
        rospy.loginfo(f"Publishing velocity - linear: {self.current_linear:.2f}, angular: {self.current_angular:.2f}")

    def stop(self):
        vel_msg = Twist()
        self.pub.publish(vel_msg)

def main():
    # Launch Gazebo with Husky
    launch_cmd = 'roslaunch husky_gazebo husky_playpen.launch world_name:=/opt/ros/noetic/share/husky_gazebo/worlds/Rowplaypen.world'
    subprocess.Popen(launch_cmd.split())
    rospy.sleep(10)  # Give more time for Gazebo to load
    
    # Launch RViz
    rviz_cmd = 'roslaunch husky_viz view_robot.launch'
    subprocess.Popen(rviz_cmd.split())
    rospy.sleep(5)
    
    # Start the controller
    controller = HuskyController()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        controller.stop()
        rospy.loginfo("Navigation stopped by user")
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
