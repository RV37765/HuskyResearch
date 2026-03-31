#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

# your map values from yaml
ORIGIN_X    = -97.695411
ORIGIN_Y    = -39.422331
RESOLUTION  = 0.05
MAP_HEIGHT  = 645

def map_to_pixel(map_x, map_y):
    pixel_x = int((map_x - ORIGIN_X) / RESOLUTION)
    pixel_y = int(MAP_HEIGHT - ((map_y - ORIGIN_Y) / RESOLUTION))
    return pixel_x, pixel_y


def get_position():
    rospy.init_node('get_position')
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



if __name__ == '__main__':
    get_position()