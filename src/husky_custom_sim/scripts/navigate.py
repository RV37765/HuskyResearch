#!/usr/bin/env python3
import time
import rospy
import tf
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# Map constants — must match hallway_strip.yaml
ORIGIN_X   = -97.695411
ORIGIN_Y   = -39.422331
RESOLUTION = 0.05
HEIGHT     = 645
WIDTH      = 1161


def pixel_to_map(pixel_x, pixel_y):
    map_x = ORIGIN_X + (pixel_x * RESOLUTION)
    map_y = ORIGIN_Y + ((HEIGHT - pixel_y) * RESOLUTION)
    return map_x, map_y


def map_to_pixel(map_x, map_y):
    pixel_x = int((map_x - ORIGIN_X) / RESOLUTION)
    pixel_y = int(HEIGHT - ((map_y - ORIGIN_Y) / RESOLUTION))
    return pixel_x, pixel_y


def wait_for_tf(listener, timeout=60):
    """Wait until AMCL publishes a valid map->base_link TF inside map bounds."""
    x_min = ORIGIN_X
    x_max = ORIGIN_X + WIDTH * RESOLUTION
    y_min = ORIGIN_Y
    y_max = ORIGIN_Y + HEIGHT * RESOLUTION

    rospy.loginfo("Waiting for AMCL to localize (position must be inside map)...")
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
            (trans, _) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
            x, y = trans[0], trans[1]

            if x_min < x < x_max and y_min < y < y_max:
                px, py = map_to_pixel(x, y)
                rospy.loginfo(f"AMCL localized: map ({x:.2f}, {y:.2f}) pixel ({px}, {py})")
                return True
            else:
                rospy.loginfo(f"  position ({x:.2f}, {y:.2f}) outside map bounds, waiting...")
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.loginfo("TF not yet available, waiting...")
        rospy.sleep(2.0)

    rospy.logwarn("AMCL did not localize within timeout.")
    return False


def send_goal_and_wait(client, x, y, goal_num):
    """Send a goal via actionlib and block until reached or failed."""
    rospy.loginfo(f"========== GOAL {goal_num} ==========")
    rospy.loginfo(f"Sending goal: map ({x:.2f}, {y:.2f})")
    px, py = map_to_pixel(x, y)
    rospy.loginfo(f"Pixel coords: ({px}, {py})")

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0

    client.send_goal(goal)
    client.wait_for_result()

    state = client.get_state()
    if state == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo(f"Goal {goal_num} REACHED!")
        return True
    else:
        rospy.logwarn(f"Goal {goal_num} FAILED (state={state})")
        return False


def main():
    rospy.init_node('reliable_navigator')
    listener = tf.TransformListener()
    rospy.sleep(1.0)

    # Wait for AMCL to localize — give a 2D pose estimate in Foxglove first
    rospy.loginfo("Make sure you have given a 2D pose estimate in Foxglove.")
    rospy.loginfo("Drive the robot slowly for a few seconds if needed.")

    if not wait_for_tf(listener, timeout=60):
        rospy.logwarn("Could not get TF — check AMCL is running and pose estimate was given")
        return

    rospy.loginfo("Waiting 3s for costmaps to populate...")
    rospy.sleep(3.0)

    # Connect to move_base action server
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base.")

    # Waypoints — pixel coords on hallway_strip.pgm
    # Robot starts at ~(1021, 234), facing down (+y direction in map)
    waypoints = [
        pixel_to_map(1021, 314),   # ~4m ahead
        pixel_to_map(1021, 394),   # ~8m ahead
    ]

    for i, (wx, wy) in enumerate(waypoints):
        success = send_goal_and_wait(client, wx, wy, i + 1)
        if not success:
            rospy.logwarn(f"Stopping at goal {i+1} — navigation failed")
            break

    rospy.loginfo("Waypoint sequence complete.")


if __name__ == '__main__':
    main()
