#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <string>

//Change these constants for vehicle kinematics
#define DEFAULT_LINEAR 0.4	     //LINEAR SPEED for no obstacle detected
#define DEFAULT_ANGULAR 0	      //ANGULAR SPEED (Turn)
#define DISTANCE 1.1		       //Maximum distance to consider point an obstacle
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


// (for now leave empty, but we can tune later)
std::vector<std::pair<int,int>> OCCLUSION_MASKS = {};

ros::Publisher pub;



//PURPOSE: Compute the minimum element contained in a vector to determine closest distance
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


void computeDirection(float front_avg, float front_left_avg, float front_right_avg,
                      float left_avg,  float right_avg)
{
    geometry_msgs::Twist cmd;
    float linearx = 0.0f, angularz = 0.0f;
    std::string case_desc;
    static int clear_count = 0;

    const float HARD_STOP_DIST = 0.8;
    const float GO_CLEARANCE   = 1.4;
    const float SIDE_CLEARANCE = .85;

    // --- 1. Emergency hard stop if anything close in the forward arc ---
    if (front_avg < HARD_STOP_DIST || front_left_avg < HARD_STOP_DIST || front_right_avg < HARD_STOP_DIST)
    {
        case_desc = "Case: HARD STOP — obstacle in front VERY CLOSE";
        linearx = 0.0;
        angularz = (left_avg > right_avg) ? +TURN_ANGULAR_SPEED : -TURN_ANGULAR_SPEED;
        g_last_turn = (left_avg > right_avg) ? +1 : -1;
        clear_count = 0;
    }
    else
    {
        // --- 2. Normal behavior: check whether anything blocks forward progress ---
        bool front_blocked = (front_avg < DISTANCE ||
                              front_left_avg < (DISTANCE) ||
                              front_right_avg < (DISTANCE);

        if (!front_blocked)
        {
            if (front_avg > GO_CLEARANCE && front_left_avg > GO_CLEARANCE && front_right_avg > GO_CLEARANCE &&
                left_avg > SIDE_CLEARANCE && right_avg > SIDE_CLEARANCE)
            {
                clear_count++;
                if (clear_count >= 7)
                {
                    case_desc = "Case: Clear ahead → move forward";
                    linearx = NEW_LINEARX;
                    angularz = 0.0;

                    // corridor-centering bias
                    if (left_avg < right_avg - 0.3)      angularz -= 0.2;
                    else if (right_avg < left_avg - 0.3) angularz += 0.2;

                    // clamp steering
                    if (angularz > 0.5)  angularz = 0.5;
                    if (angularz < -0.5) angularz = -0.5;
                }
                else
                {
                    case_desc = "Case: Recently cleared → keep last turn";
                    linearx = 0.0;
                    angularz = g_last_turn * TURN_ANGULAR_SPEED;
                }
            }
            else
            {
                case_desc = "Case: Partial clearance → keep turning";
                linearx = 0.0;
                angularz = g_last_turn * TURN_ANGULAR_SPEED;
                clear_count = 0;
            }
        }
        else
        {
            // --- 3. Obstacle avoidance: pick the side with more diagonal clearance ---
            clear_count = 0;
            if (front_left_avg > front_right_avg)
            {
                case_desc = "Case: Front blocked → turn left (more space)";
                angularz = +TURN_ANGULAR_SPEED;
                g_last_turn = +1;
            }
            else
            {
                case_desc = "Case: Front blocked → turn right (more space)";
                angularz = -TURN_ANGULAR_SPEED;
                g_last_turn = -1;
            }
            linearx = 0.0;
        }
    }

    // --- 4. Publish the command ---
    cmd.linear.x  = linearx;
    cmd.angular.z = angularz;
    pub.publish(cmd);

    ROS_INFO("%s (F=%.2f, FL=%.2f, FR=%.2f, L=%.2f, R=%.2f)",
             case_desc.c_str(), front_avg, front_left_avg, front_right_avg, left_avg, right_avg);
}


//------Processing and splitting data-------
//front, front-left, front-right, left, right
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
    auto idx_from_angle = [&](double angle_rad){
			    if (angle_rad < msg->angle_min) angle_rad = msg->angle_min;
			    if (angle_rad > msg->angle_max) angle_rad = msg->angle_max;
			    return (int)((angle_rad - msg->angle_min) / msg->angle_increment);
			  };

 
    // LiDAR flipped 180° → reverse all signs
    int front_idx        = idx_from_angle(0.0);               // now forward points backward
    int front_left_idx   = idx_from_angle(M_PI / (180.0 * 35.0));
    int front_right_idx  = idx_from_angle(-M_PI / (180.0 * 35.0));
    int left_idx         = idx_from_angle(M_PI / 2.0);         // stays same
    int right_idx        = idx_from_angle(-M_PI / 2.0);        // stays same


   // --- Compute 5 averaged distances ---
    float front_avg        = avg_window(front_idx,       FRONT_WINDOW_DEG);
    float front_left_avg   = avg_window(front_left_idx,  FRONT_WINDOW_DEG);
    float front_right_avg  = avg_window(front_right_idx, FRONT_WINDOW_DEG);
    float left_avg         = avg_window(left_idx,        SIDE_WINDOW_DEG);
    float right_avg        = avg_window(right_idx,       SIDE_WINDOW_DEG);


    // Keep your back-left/right mins if you still log them; not needed for decisions now
    std::vector<float> backLeft, backRight;
    int leftLimit  = BACK_ANGLE_PROPORTION_THRESHOLD * size;
    int rightLimit = size - leftLimit;
    for (int i = 0; i <= leftLimit; ++i) if (valid(i)) backLeft.push_back(ranges[i]);
    for (int i = rightLimit; i < size; ++i) if (valid(i)) backRight.push_back(ranges[i]);
    float minBackLeft  = backLeft.empty()  ? msg->range_max : min_element(backLeft);
    float minBackRight = backRight.empty() ? msg->range_max : min_element(backRight);

    // Decide and publish
    computeDirection(front_avg, front_left_avg, front_right_avg, left_avg, right_avg);

    ROS_INFO("SIZE: %d", size);
    ROS_INFO("avg range on front: %.3f", front_avg);
    ROS_INFO("avg range on front-right: %.3f", front_right_avg);
    ROS_INFO("avg range on front-left: %.3f", front_left_avg);
    ROS_INFO("avg range on left:  %.3f", left_avg);
    ROS_INFO("avg range on right: %.3f", right_avg);
   
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
