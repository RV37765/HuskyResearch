#!/usr/bin/env python3
import rospy
import time
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from std_srvs.srv import Empty

# Map constants — must match hallway_strip.yaml
ORIGIN_X   = -97.695411
ORIGIN_Y   = -39.422331
RESOLUTION = 0.05
HEIGHT     = 645


def pixel_to_map(pixel_x, pixel_y):
    map_x = ORIGIN_X + (pixel_x * RESOLUTION)
    map_y = ORIGIN_Y + ((HEIGHT - pixel_y) * RESOLUTION)
    return map_x, map_y


def map_to_pixel(map_x, map_y):
    pixel_x = int((map_x - ORIGIN_X) / RESOLUTION)
    pixel_y = int(HEIGHT - ((map_y - ORIGIN_Y) / RESOLUTION))
    return pixel_x, pixel_y


def get_position():
    rospy.loginfo("Reading current AMCL position...")
    msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout=10)

    map_x = msg.pose.pose.position.x
    map_y = msg.pose.pose.position.y
    pixel_x, pixel_y = map_to_pixel(map_x, map_y)
    uncertainty = msg.pose.covariance[0] + msg.pose.covariance[7]

    rospy.loginfo("========== ROBOT POSITION ==========")
    rospy.loginfo(f"Map coords:   x={map_x:.2f}  y={map_y:.2f}")
    rospy.loginfo(f"Pixel coords: x={pixel_x}  y={pixel_y}")
    rospy.loginfo(f"Uncertainty:  {uncertainty:.5f}  (lower is better)")
    rospy.loginfo("====================================")
    return map_x, map_y


def trigger_global_localization():
    rospy.loginfo("Spreading AMCL particles across map...")
    rospy.wait_for_service('/global_localization', timeout=10)
    rospy.ServiceProxy('/global_localization', Empty)()
    rospy.loginfo("Global localization triggered.")


def rotate_to_localize(duration=25, speed=0.3):
    rospy.loginfo(f"Rotating {duration}s to help AMCL converge...")
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.sleep(1)  # wait for publisher to connect

    twist = Twist()
    twist.angular.z = speed

    start = time.time()
    rate = rospy.Rate(10)
    while time.time() - start < duration:
        pub.publish(twist)
        rate.sleep()

    pub.publish(Twist())  # stop
    rospy.loginfo("Rotation complete.")


def wait_for_convergence(threshold=0.05, timeout=60):
    """
    Wait until AMCL position uncertainty drops below threshold.
    Uncertainty = sum of x and y variance from the covariance matrix.
    Typical values: < 0.01 after /initialpose, < 0.05 after global_localization.
    """
    rospy.loginfo(f"Waiting for AMCL convergence (uncertainty < {threshold})...")
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout=5)
            uncertainty = msg.pose.covariance[0] + msg.pose.covariance[7]
            rospy.loginfo(f"  uncertainty: {uncertainty:.5f}")
            if uncertainty < threshold:
                rospy.loginfo("AMCL converged.")
                return True
        except rospy.ROSException:
            rospy.logwarn("No AMCL message received, retrying...")

    rospy.logwarn("AMCL did not converge within timeout.")
    return False


def send_goal(x, y):
    rospy.loginfo(f"Sending goal: map ({x:.2f}, {y:.2f})")
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
    rospy.sleep(0.5)  # wait for publisher to connect

    goal = PoseStamped()
    goal.header.frame_id = 'map'
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.w = 1.0

    pub.publish(goal)


def main():
    rospy.init_node('reliable_navigator')

    # Step 1 — spread AMCL particles across the full map
    trigger_global_localization()
    rospy.sleep(2)

    # Step 2 — rotate in place so AMCL sees scan variation from different angles
    rotate_to_localize(duration=25, speed=0.3)

    # Step 3 — wait for particles to converge to a confident estimate
    if not wait_for_convergence(threshold=0.05, timeout=60):
        rospy.logwarn("Could not localize — drive manually then retry")
        return

    # Step 4 — log confirmed position before moving
    get_position()

    # Step 5 — send goal
    # Edit pixel coordinates to match your target on hallway_strip.pgm
    map_x, map_y = pixel_to_map(681, 377)
    send_goal(map_x, map_y)


if __name__ == '__main__':
    main()
