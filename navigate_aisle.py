#!/usr/bin/env python3
import rospy
import actionlib
import json
import sys
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

WAYPOINTS_FILE = '/home/administrator/HuskyRealWork/waypoints.json'

def load_waypoints(path):
    with open(path) as f:
        return json.load(f)

def send_goal(client, x, y):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id    = "map"
    goal.target_pose.header.stamp       = rospy.Time.now()
    goal.target_pose.pose.position.x    = x
    goal.target_pose.pose.position.y    = y
    goal.target_pose.pose.orientation.w = 1.0

    rospy.loginfo(f"Going to x={x:.2f} y={y:.2f}")
    client.send_goal(goal)
    client.wait_for_result()

    return client.get_state() == actionlib.GoalStatus.SUCCEEDED

def go_to_aisle(aisle_id):
    data = load_waypoints(WAYPOINTS_FILE)

    if aisle_id not in data:
        rospy.logerr(f"Aisle '{aisle_id}' not found in waypoints.json")
        rospy.loginfo(f"Available aisles: {list(data.keys())}")
        return

    aisle     = data[aisle_id]
    waypoints = aisle["waypoints"]

    rospy.loginfo(f"Navigating {aisle_id} — {len(waypoints)} waypoints")

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base...")
    client.wait_for_server()

    for i, (wx, wy) in enumerate(waypoints):
        rospy.loginfo(f"Waypoint {i+1}/{len(waypoints)}")
        success = send_goal(client, wx, wy)

        if success:
            rospy.loginfo(f"Waypoint {i+1} reached")
        else:
            rospy.logwarn(f"Waypoint {i+1} failed — skipping")

    rospy.loginfo(f"Finished {aisle_id}")

def list_aisles():
    data = load_waypoints(WAYPOINTS_FILE)
    print("\nAvailable aisles:")
    for aisle_id, aisle_data in data.items():
        wp_count = len(aisle_data["waypoints"])
        entry    = aisle_data["entry"]
        print(f"  {aisle_id}: {wp_count} waypoints | entry: x={entry[0]:.2f} y={entry[1]:.2f}")

if __name__ == '__main__':
    rospy.init_node('aisle_navigator')

    # list available aisles
    list_aisles()

    # get aisle from command line argument
    # usage: python3 navigate_aisle.py aisle_1
    if len(sys.argv) > 1:
        target = sys.argv[1]
        rospy.loginfo(f"Target aisle: {target}")
        go_to_aisle(target)
    else:
        rospy.logwarn("No aisle specified")
        rospy.loginfo("Usage: python3 navigate_aisle.py aisle_1")