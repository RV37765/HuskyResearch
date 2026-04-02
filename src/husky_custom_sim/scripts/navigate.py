#!/usr/bin/env python3
import rospy
import time
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

# Map constants — must match hallway_map_slam_test.yaml
ORIGIN_X   = -47.875022
ORIGIN_Y   = -2.997483
RESOLUTION = 0.05
HEIGHT     = 1097


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


def set_initial_pose(x, y, yaw):
    """
    Publish a known starting pose to /initialpose.
    This is more reliable than global_localization in symmetric corridors
    because it gives AMCL both position AND orientation, eliminating the
    180-degree heading ambiguity that causes reversed navigation.

    yaw is in radians (0 = facing map +x, pi = facing map -x).
    Use pixel_to_map() to get x, y from the PGM image.
    """
    import math
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
    rospy.sleep(0.5)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = rospy.Time.now()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
    msg.pose.covariance[0]  = 0.25   # x uncertainty (0.5m std dev)
    msg.pose.covariance[7]  = 0.25   # y uncertainty
    msg.pose.covariance[35] = 0.068  # yaw uncertainty (~15 degrees)

    pub.publish(msg)
    rospy.loginfo(f"Initial pose set: map ({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f} deg")


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

    # Step 1 — set known starting pose (position + orientation).
    # Place the robot at the starting pixel on hallway_strip.pgm, physically
    # facing toward the goal, then run this script.
    #
    # Starting pixel: (965, 257) — right side of corridor near lab entrance
    # Facing toward goal (681, 377): yaw = -2.74 rad (~-157 degrees from +x)
    #
    # To change starting location: update the pixel coordinates below.
    # To change facing direction: yaw=0 faces map +x, yaw=3.14 faces map -x.
    # Start pixel (914, 981) on hallway_map_slam_test.pgm
    # Facing toward goal (661, 810): yaw = 2.55 rad (~146 degrees from +x)
    start_x, start_y = pixel_to_map(914, 981)
    set_initial_pose(start_x, start_y, yaw=2.55)
    rospy.sleep(2)

    # Step 2 — wait for slam_toolbox to converge from the pose hint
    if not wait_for_convergence(threshold=0.05, timeout=30):
        rospy.logwarn("Could not localize — check that robot is at the expected starting position")
        return

    # Step 3 — log confirmed position before moving
    get_position()

    # Step 4 — send goal
    # Goal pixel (661, 810) on hallway_map_slam_test.pgm
    map_x, map_y = pixel_to_map(661, 810)
    send_goal(map_x, map_y)


if __name__ == '__main__':
    main()
