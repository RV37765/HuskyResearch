#!/usr/bin/env python3
import rospy
import subprocess
import time
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

# nav = subprocess.Popen("roslaunch husky_custom_sim husky_real_nav.launch", shell=True)
# time.sleep(10)  # wait for everything to start

ORIGIN_X    = -97.695411
ORIGIN_Y    = -39.422331
RESOLUTION  = 0.05
HEIGHT  = 645

def pixel_to_map(pixel_x, pixel_y):
    map_x = ORIGIN_X + (pixel_x * RESOLUTION)
    map_y = ORIGIN_Y + ((HEIGHT - pixel_y) * RESOLUTION)
    return map_x, map_y

def map_to_pixel(map_x, map_y):
    pixel_x = int((map_x - ORIGIN_X) / RESOLUTION)
    pixel_y = int(HEIGHT - ((map_y - ORIGIN_Y) / RESOLUTION))
    return pixel_x, pixel_y


def get_position():
    #rospy.init_node('get_position')
    rospy.loginfo("Waiting for amcl pose...")
    
    msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout=10)
    
    # map coordinates
    map_x = msg.pose.pose.position.x
    map_y = msg.pose.pose.position.y
    
    # pixel coordinates
    pixel_x, pixel_y = map_to_pixel(map_x, map_y)
    
    # covariance — lower means more confident
    cov = msg.pose.covariance
    uncertainty = cov[0] + cov[7]
    
    rospy.loginfo("========== ROBOT POSITION ==========")
    rospy.loginfo(f"Map coords:   x={map_x:.2f}  y={map_y:.2f}")
    rospy.loginfo(f"Pixel coords: x={pixel_x}  y={pixel_y}")
    rospy.loginfo(f"Uncertainty:  {uncertainty:.3f} (lower is better)")
    rospy.loginfo("====================================")
    
    return map_x, map_y, pixel_x, pixel_y


def wait_for_localization(timeout=60):
    rospy.loginfo("Waiting for amcl to localize...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                "rostopic echo /amcl_pose -n 1",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout:
                rospy.loginfo("AMCL localized successfully")
                return True
        except subprocess.TimeoutExpired:
            pass
        
        rospy.loginfo("Not localized yet, waiting...")
        time.sleep(3)
    
    rospy.logwarn("AMCL failed to localize within timeout")
    return False

def set_wide_initial_pose():
    rospy.loginfo("Setting wide initial pose for global localization...")
    subprocess.run("""rostopic pub /initialpose geometry_msgs/PoseWithCovarianceStamped "header:
  frame_id: 'map'
pose:
  pose:
    position:
      x: 0.0
      y: 0.0
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
  covariance: [9.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 9.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 3.14]" --once""", shell=True)

def trigger_global_localization():
    rospy.loginfo("Triggering global localization...")
    subprocess.run("rosservice call /global_localization '{}'", shell=True)

def send_goal(x, y):
    rospy.loginfo(f"Sending goal x={x} y={y}")
    subprocess.run(f"""rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "header:
  frame_id: 'map'
pose:
  position:
    x: {x}
    y: {y}
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0" --once""", shell=True)

def main():
    rospy.init_node('reliable_navigator')

    # Step 1 - set wide initial pose
    set_wide_initial_pose()
    time.sleep(2)

    # Step 2 - trigger global localization
    trigger_global_localization()
    time.sleep(2)

    # Step 3 - rotate in place to help amcl localize
    rospy.loginfo("Rotating to help amcl localize...")
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.sleep(1)  # wait for publisher to connect

    twist = Twist()
    twist.angular.z = 0.3  # slow rotation

    start = time.time()
    rate = rospy.Rate(10)
    while time.time() - start < 20:
        pub.publish(twist)
        rate.sleep()

    # stop rotation
    pub.publish(Twist())
    rospy.loginfo("Rotation complete")

    # Step 4 - wait for localization
    if wait_for_localization(timeout=30):
        get_position()
        # Step 5 - send goal
        ##############################coordinates send here (pixel coordinates based on pgm file)
        map_x,map_y = pixel_to_map(681,377)
        send_goal(map_x, map_y)
    else:
        rospy.logwarn("Could not localize - try driving manually first")

if __name__ == '__main__':
    main()