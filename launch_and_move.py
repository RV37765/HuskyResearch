#!/usr/bin/env python
import rospy
import subprocess
import time
from geometry_msgs.msg import Twist

def move_husky():
    rospy.init_node('husky_mover')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()
    
    # Wait for Gazebo to fully start
    time.sleep(10)
    
    # Move forward
    vel_msg.linear.x = 0.5
    for _ in range(90):
        pub.publish(vel_msg)
        time.sleep(0.1)
    
    # Stop
    vel_msg.linear.x = 0
    pub.publish(vel_msg)

if __name__ == '__main__':
    # Launch Gazebo with Husky
    subprocess.Popen(['roslaunch', 'husky_gazebo', 'empty_world.launch'])
    
    # Start movement script
    try:
        move_husky()
    except rospy.ROSInterruptException:
        pass
