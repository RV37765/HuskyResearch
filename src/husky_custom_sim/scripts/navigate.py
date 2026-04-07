#!/usr/bin/env python3
import math
import time
import rospy
import tf
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


def get_position(listener):
    """Read robot position from TF (map -> base_link). Works with both AMCL and slam_toolbox."""
    try:
        listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(5.0))
        (trans, rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        map_x, map_y = trans[0], trans[1]
        pixel_x, pixel_y = map_to_pixel(map_x, map_y)
        rospy.loginfo("========== ROBOT POSITION ==========")
        rospy.loginfo(f"Map coords:   x={map_x:.2f}  y={map_y:.2f}")
        rospy.loginfo(f"Pixel coords: x={pixel_x}  y={pixel_y}")
        rospy.loginfo("====================================")
        return map_x, map_y
    except (tf.Exception, tf.LookupException, tf.ConnectivityException) as e:
        rospy.logwarn(f"Could not get position from TF: {e}")
        return None, None


def set_initial_pose(x, y, yaw):
    """
    Publish a known starting pose to /initialpose.
    slam_toolbox listens to this topic just like AMCL does.
    yaw in radians: 0 = facing map +x, pi = facing map -x.
    """
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


def wait_for_tf(listener, timeout=60):
    """
    Wait until slam_toolbox publishes a valid map->base_link TF with a position
    that is actually inside the map bounds. A TF that exists but places the robot
    outside the map means slam_toolbox hasn't localized yet.
    """
    # Map bounds derived from origin + dimensions
    x_min = ORIGIN_X
    x_max = ORIGIN_X + 1265 * RESOLUTION
    y_min = ORIGIN_Y
    y_max = ORIGIN_Y + HEIGHT * RESOLUTION

    rospy.loginfo("Waiting for slam_toolbox to localize (position must be inside map)...")
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
            (trans, _) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
            x, y = trans[0], trans[1]

            if x_min < x < x_max and y_min < y < y_max:
                rospy.loginfo(f"slam_toolbox localized: map ({x:.2f}, {y:.2f})")
                return True
            else:
                rospy.loginfo(f"  position ({x:.2f}, {y:.2f}) outside map bounds — not yet localized, keep driving...")
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.loginfo("TF not yet available, waiting...")
        rospy.sleep(2.0)

    rospy.logwarn("slam_toolbox did not localize within timeout.")
    return False


def send_goal(x, y):
    rospy.loginfo(f"Sending goal: map ({x:.2f}, {y:.2f})")
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
    rospy.sleep(0.5)

    goal = PoseStamped()
    goal.header.frame_id = 'map'
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.w = 1.0

    pub.publish(goal)


def main():
    rospy.init_node('reliable_navigator')
    listener = tf.TransformListener()
    rospy.sleep(1.0)  # give TF listener time to fill its buffer

    # Step 1 — drive manually with PS4 for 15-20 seconds before running this script.
    # slam_toolbox will localize from scan matching as the robot moves through the space.
    # No initial pose hint needed — self-localization test.
    rospy.loginfo("Waiting for slam_toolbox to self-localize...")
    rospy.loginfo("Drive the robot manually with PS4 for 15-20 seconds, then wait.")

    # Step 2 — wait for slam_toolbox to publish TF from scan matching
    if not wait_for_tf(listener, timeout=30):
        rospy.logwarn("Could not get TF — check slam_toolbox is running")
        return

    # Step 3 — log confirmed position
    get_position(listener)

    # Brief pause — gives local costmap time to populate with scan data
    # before the goal is sent. Without this, the local planner receives the
    # global path before it has enough data to transform it.
    rospy.loginfo("Waiting 3s for costmaps to populate...")
    rospy.sleep(3.0)

    # Step 4 — send goal
    # Goal pixel (984, 120) on hallway_map_slam_test.pgm
    map_x, map_y = pixel_to_map(984, 120)
    send_goal(map_x, map_y)


if __name__ == '__main__':
    main()
