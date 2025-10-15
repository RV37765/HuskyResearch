#!/usr/bin/env python3
import rospy, subprocess, signal
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

proc = None
pub = None

# PS4 buttons (check with `rostopic echo /joy` if indices differ):
# 6 = Share, 7 = Options
BUTTON_SHARE = 6
BUTTON_OPTIONS = 7

def joy_callback(msg):
    global proc, pub

    if msg.buttons[BUTTON_OPTIONS] == 1 and proc is None:
        rospy.loginfo("Launching obstacle avoidance...")
        proc = subprocess.Popen(["roslaunch", "obstacle_avoidance", "obstacle_avoidance_real.launch"])

    if msg.buttons[BUTTON_SHARE] == 1 and proc is not None:
        rospy.loginfo("Stopping obstacle avoidance...")
        proc.send_signal(signal.SIGINT)  # Ctrl-C to roslaunch
        proc = None

        # Stop robot motion
        twist = Twist()
        pub.publish(twist)

def main():
    global pub
    rospy.init_node("ps4_obstacle_trigger")
    rospy.Subscriber("/joy_teleop/joy", Joy, joy_callback)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.spin()

if __name__ == "__main__":
    main()
