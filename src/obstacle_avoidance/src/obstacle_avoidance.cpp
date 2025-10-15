#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <string>

//Change these constants for vehicle kinematics
#define DEFAULT_LINEAR 0.4	     //LINEAR SPEED for no obstacle detected
#define DEFAULT_ANGULAR 0	      //ANGULAR SPEED (Turn)
#define DISTANCE 1.2		       //Maximum distance to consider point an obstacle
#define NEW_LINEARX 0.3	       //
#define TURN_ANGULAR_SPEED 0.75	   //Turn speed
#define MINIMUM_DISTANCE_THRESHOLD 0.1 //how sensitive LiDAR is to small distance values (DEFAULT: 0.1)
#define BACK_ANGLE_PROPORTION_THRESHOLD 0.33 //0.24 = 60 deg / 270 , 0.33 = 90 deg / 270

#define FRONT_WINDOW_DEG 17.5            // averaging window around front (± degrees)
#define SIDE_WINDOW_DEG 35             // averaging window for side look

#define REVERSE_THRESHOLD 0.8         // if front < this, consider stuck
#define STUCK_CYCLES 15                // how many consecutive cycles before escape
#define REVERSE_TIME 0.8               // sec to back up during escape
#define TURN_TIME 0.8                  // sec to turn during escape

// ---- Global state for stuck detection ----
int g_stuck_count = 0;          // Counts how many cycles robot appears stuck
int g_last_turn   = 1;          // Remember last turn direction (+1 left, -1 right)
int g_clear_count = 0;   // counts consecutive clear cycles before moving forward

enum Mode { NORMAL, ESCAPE_REVERSE, ESCAPE_TURN };
Mode g_mode = NORMAL;
ros::Time g_escape_end;


// Optional: mask indices if your real Husky Velodyne mount causes occlusions
// (for now leave empty, but we can tune later)
std::vector<std::pair<int,int>> OCCLUSION_MASKS = {};

ros::Publisher pub;
//HELPER FUNCTION
//PURPOSE: Compute the minimum element contained in a vector to determine
// the minimum distance to the closest obstacle in the given partition of sensor.
float min_element(std::vector<float> first)
{
	float min = 50.0;

	for (std::vector<float>::iterator it = first.begin(); it < first.end(); it++)
	{
		if (*it<min && * it> MINIMUM_DISTANCE_THRESHOLD)
		{
			min = *it;
		}
	}
	return min;
}

//HELPER FUNCTION
//PURPOSE: Compute the direction based on where an obstacle is detected.
//Function publishes a message with new direction, linear, and angular velocity.
//IDEA: New angle can be dependent on how close the object is (inverse sigmoid)
// void computeDirection(float front_avg, float left_avg, float right_avg)
// {
//     geometry_msgs::Twist cmd;
//     float linearx = 0.0f, angularz = 0.0f;
//     std::string case_desc;

//     // Static counter for linger logic
//     static int clear_count = 0;

//     // Tunable thresholds
//     const float HARD_STOP_DIST = 0.8;    // emergency stop
//     const float GO_CLEARANCE   = 2.0;    // clearance needed before moving
//     const float SIDE_CLEARANCE = 1.2;    // side clearance needed before moving

//     bool front_blocked = (front_avg < DISTANCE);

//     // --- Safety hard stop ---
//     if (front_avg < HARD_STOP_DIST) {
//         case_desc = "Case: HARD STOP - obstacle directly ahead!";
//         linearx = 0.0;
//         angularz = (left_avg > right_avg) ? +TURN_ANGULAR_SPEED : -TURN_ANGULAR_SPEED;
//         g_last_turn = (left_avg > right_avg) ? +1 : -1;
//         clear_count = 0;  // reset
//         cmd.linear.x = linearx;
//         cmd.angular.z = angularz;
//         pub.publish(cmd);
//         ROS_WARN("%s (front=%.2f)", case_desc.c_str(), front_avg);
//         return;
//     }

//     // --- Normal behavior ---
//     if (!front_blocked) {
//         // Require both front AND side clearance before going
//         if (front_avg > GO_CLEARANCE && left_avg > SIDE_CLEARANCE && right_avg > SIDE_CLEARANCE) {
//             clear_count++;
//             if (clear_count >= 5) {   // ~0.5s of consistent clearance at 10 Hz
//                 case_desc = "Case: Clear ahead → go";
//                 linearx = NEW_LINEARX;
//                 angularz = 0.0;
//             } else {
//                 case_desc = "Case: Recently cleared → keep turning";
//                 linearx = 0.0;
//                 angularz = g_last_turn * TURN_ANGULAR_SPEED;
//             }
//         } else {
//             clear_count = 0;
//             case_desc = "Case: Clearance insufficient → keep turning";
//             linearx = 0.0;
//             angularz = g_last_turn * TURN_ANGULAR_SPEED;
//         }
//     } else {
//         // Front is blocked
//         clear_count = 0;
//         if (left_avg > right_avg) {
//             case_desc = "Case: Front blocked → turn left in place";
//             linearx = 0.0;
//             angularz = +TURN_ANGULAR_SPEED;
//             g_last_turn = +1;
//         } else if (right_avg > left_avg) {
//             case_desc = "Case: Front blocked → turn right in place";
//             linearx = 0.0;
//             angularz = -TURN_ANGULAR_SPEED;
//             g_last_turn = -1;
//         } else {
//             case_desc = "Case: Front blocked & tie → keep last turn in place";
//             linearx = 0.0;
//             angularz = g_last_turn * TURN_ANGULAR_SPEED;
//         }
//     }

//     cmd.linear.x = linearx;
//     cmd.angular.z = angularz;
//     pub.publish(cmd);
//     ROS_INFO("%s (front=%.2f left=%.2f right=%.2f)", case_desc.c_str(), front_avg, left_avg, right_avg);
// }

void computeDirection(float front_avg, float left_avg, float right_avg)
{
    geometry_msgs::Twist cmd;
    float linearx = 0.0f, angularz = 0.0f;
    std::string case_desc;

    static int clear_count = 0;

    const float HARD_STOP_DIST = 0.8;    // emergency stop
    const float GO_CLEARANCE   = 2.0;    // forward clearance needed
    const float SIDE_CLEARANCE = 1.2;    // side clearance needed

    bool front_blocked = (front_avg < DISTANCE);

    // --- Safety hard stop ---
    if (front_avg < HARD_STOP_DIST) {
        case_desc = "Case: HARD STOP - obstacle directly ahead!";
        linearx = 0.0;
        angularz = (left_avg > right_avg) ? +TURN_ANGULAR_SPEED : -TURN_ANGULAR_SPEED;
        g_last_turn = (left_avg > right_avg) ? +1 : -1;
        clear_count = 0;
        cmd.linear.x = linearx;
        cmd.angular.z = angularz;
        pub.publish(cmd);
        ROS_WARN("%s (front=%.2f)", case_desc.c_str(), front_avg);
        return;
    }

    // --- Normal behavior ---
    if (!front_blocked) {
        if (front_avg > GO_CLEARANCE && left_avg > SIDE_CLEARANCE && right_avg > SIDE_CLEARANCE) {
            clear_count++;
            if (clear_count >= 5) {
                case_desc = "Case: Clear ahead → go";

                // Base forward command
                linearx = NEW_LINEARX;
                angularz = 0.0;

                // Corridor-centering bias
                float corridor_bias = 0.0;
                if (left_avg < right_avg - 0.3) {
                    corridor_bias = -0.2;   // too close left wall → steer right
                } else if (right_avg < left_avg - 0.3) {
                    corridor_bias = +0.2;   // too close right wall → steer left
                }
                angularz += corridor_bias;

                // Limit turn bias so it doesn't fight forward motion too hard
                if (angularz > 0.5) angularz = 0.5;
                if (angularz < -0.5) angularz = -0.5;

            } else {
                case_desc = "Case: Recently cleared → keep turning";
                linearx = 0.0;
                angularz = g_last_turn * TURN_ANGULAR_SPEED;
            }
        } else {
            clear_count = 0;
            case_desc = "Case: Clearance insufficient → keep turning";
            linearx = 0.0;
            angularz = g_last_turn * TURN_ANGULAR_SPEED;
        }
    } else {
        // Front is blocked
        clear_count = 0;
        if (left_avg > right_avg) {
            case_desc = "Case: Front blocked → turn left in place";
            linearx = 0.0;
            angularz = +TURN_ANGULAR_SPEED;
            g_last_turn = +1;
        } else if (right_avg > left_avg) {
            case_desc = "Case: Front blocked → turn right in place";
            linearx = 0.0;
            angularz = -TURN_ANGULAR_SPEED;
            g_last_turn = -1;
        } else {
            case_desc = "Case: Front blocked & tie → keep last turn in place";
            linearx = 0.0;
            angularz = g_last_turn * TURN_ANGULAR_SPEED;
        }
    }

    cmd.linear.x = linearx;
    cmd.angular.z = angularz;
    pub.publish(cmd);
    ROS_INFO("%s (front=%.2f left=%.2f right=%.2f)", case_desc.c_str(), front_avg, left_avg, right_avg);
}

//Callback Function To Process Lidar Data and Partition Into 5 Arrays:
//Left, FrontLeft, Front, FrontRight, and Right
//720 total laser scans divided by 5 = 144 scans for each partition.
//Introduce clustering algorithm that passes angle ranges of detected obstacles...
void laserCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
    const auto &ranges = msg->ranges;
    int size = ranges.size();
    if (size == 0) return;

    auto valid = [&](int idx) {
        if (idx < 0 || idx >= size) return false;
        // occlusion mask filter
        for (auto &m : OCCLUSION_MASKS) {
            if (idx >= m.first && idx <= m.second) return false;
        }
        float r = ranges[idx];
        return (r > MINIMUM_DISTANCE_THRESHOLD && r < msg->range_max && !std::isnan(r) && !std::isinf(r));
    };

    auto avg_window = [&](int center_idx, int window_deg)->float {
        // convert degrees to samples using angle_increment (rad)
        int window = std::max(1, (int)std::round((window_deg * M_PI/180.0) / msg->angle_increment));
        double sum = 0.0; int count = 0;
        for (int i = center_idx - window; i <= center_idx + window; ++i) {
            if (valid(i)) { sum += ranges[i]; ++count; }
        }
        return (count > 0) ? (float)(sum / count) : msg->range_max;
    };

    // Index helpers: center/front is pi/2 ahead if 0 is -pi; simpler: use half for front
    // int front_idx = size / 2;                     // assumes angle_min ~ -pi, angle_max ~ +pi
    // int left_idx  = (int)std::round(size * 0.75); // +90 deg
    // int right_idx = (int)std::round(size * 0.25); // -90 deg
    auto idx_from_angle = [&](double angle_rad){
			    if (angle_rad < msg->angle_min) angle_rad = msg->angle_min;
			    if (angle_rad > msg->angle_max) angle_rad = msg->angle_max;
			    return (int)((angle_rad - msg->angle_min) / msg->angle_increment);
			  };

    // front = 0 rad
    int front_idx = idx_from_angle(0.0);
    // left = +90 deg
    int left_idx  = idx_from_angle(M_PI/2);
    // right = -90 deg
    int right_idx = idx_from_angle(-M_PI/2);



    
    float front_avg = avg_window(front_idx, FRONT_WINDOW_DEG);
    float left_avg  = avg_window(left_idx,  SIDE_WINDOW_DEG);
    float right_avg = avg_window(right_idx, SIDE_WINDOW_DEG);

    // Keep your back-left/right mins if you still log them; not needed for decisions now
    // (You can delete these two blocks if you prefer pure avg-based logic)
    std::vector<float> backLeft, backRight;
    int leftLimit  = BACK_ANGLE_PROPORTION_THRESHOLD * size;
    int rightLimit = size - leftLimit;
    for (int i = 0; i <= leftLimit; ++i) if (valid(i)) backLeft.push_back(ranges[i]);
    for (int i = rightLimit; i < size; ++i) if (valid(i)) backRight.push_back(ranges[i]);
    float minBackLeft  = backLeft.empty()  ? msg->range_max : min_element(backLeft);
    float minBackRight = backRight.empty() ? msg->range_max : min_element(backRight);

    // Decide and publish
    computeDirection(front_avg, left_avg, right_avg);

    ROS_INFO("SIZE: %d", size);
    ROS_INFO("avg range on front: %.3f", front_avg);
    ROS_INFO("avg range on left:  %.3f", left_avg);
    ROS_INFO("avg range on right: %.3f", right_avg);
    ROS_INFO("min range on back-left: %.3f",  minBackLeft);
    ROS_INFO("min range on back-right: %.3f", minBackRight);
}



int main(int argc, char **argv)
{

	ros::init(argc, argv, "laser_avoidance");
	ros::NodeHandle nh;
	pub = nh.advertise<geometry_msgs::Twist>("/husky_velocity_controller/cmd_vel", 100);
	ros::Subscriber sub = nh.subscribe("/scan", 10, laserCallback);

	ros::spin();
	return 0;
}
