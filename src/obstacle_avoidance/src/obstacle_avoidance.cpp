#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <string>

// --- Parameters (loaded from YAML via ROS Parameter Server) ---
// These replace the old #define constants. Values are set in main() from the parameter server,
// with sensible defaults if the YAML file is missing.
double linear_speed;
double turn_speed;
double hard_stop_distance;
double min_distance_threshold;
double direct_front_deg;
double front_window_deg;
double side_window_deg;

ros::Publisher pub;

void computeDirection(float front_avg, float front_left_avg, float front_right_avg,
                      float left_avg,  float right_avg)
{
    geometry_msgs::Twist cmd;
    float linearx = 0.0f, angularz = 0.0f;
    std::string case_desc;

    // --- 1. Emergency hard stop if anything close in the forward arc ---
    if (front_avg < hard_stop_distance || front_left_avg < hard_stop_distance || front_right_avg < hard_stop_distance)
    {
        if(front_avg > hard_stop_distance && abs(front_left_avg-front_right_avg) < .1){
            case_desc = "Case: Hard Stop -->  -- FRONT OPEN SIDES EQUAL ---";
            linearx = linear_speed;
            angularz = 0.0;
        } else {
            case_desc = "Case: Hard Stop —> --- FRONT BLOCKED AND SIDES UNEQUAL---";
            linearx = 0.0;
            angularz = (front_left_avg > front_right_avg) ? +turn_speed : -turn_speed;
        }

    }
    else
    {
        case_desc = "Case: Clear ahead -> ---MOVE FORWARD---";
        linearx = linear_speed;
        angularz = 0.0;
    }


    // --- 4. Publish the command ---
    cmd.linear.x  = linearx;
    cmd.angular.z = angularz;
    pub.publish(cmd);

    ROS_INFO("%s (F=%.2f, FL=%.2f, FR=%.2f)",
             case_desc.c_str(), front_avg, front_left_avg, front_right_avg);
}


//------Processing and splitting data-------
//front, front-left, front-right
void laserCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
    const auto &ranges = msg->ranges;
    int size = ranges.size();
    if (size == 0) return;

    static int iteration_count = 0;
    iteration_count++;


    auto valid = [&](int idx) {
        if (idx < 0 || idx >= size) return false;
        float r = ranges[idx];
        return (r > min_distance_threshold && r < msg->range_max && !std::isnan(r) && !std::isinf(r));
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

    auto idx_from_angle = [&](double angle_rad){
			    if (angle_rad < msg->angle_min) angle_rad = msg->angle_min;
			    if (angle_rad > msg->angle_max) angle_rad = msg->angle_max;
			    return (int)((angle_rad - msg->angle_min) / msg->angle_increment);
			  };



    int front_idx        = idx_from_angle(0.0);
    int front_left_idx   = idx_from_angle((-M_PI  * 55) / 180);
    int front_right_idx  = idx_from_angle((M_PI * 55)/ 180);
    int left_idx         = idx_from_angle(-M_PI / 2.0);
    int right_idx        = idx_from_angle(M_PI / 2.0);


    // Determines width of each window, based on center angle
    float front_avg        = avg_window(front_idx,       direct_front_deg);
    float front_left_avg   = avg_window(front_left_idx,  front_window_deg);
    float front_right_avg  = avg_window(front_right_idx, front_window_deg);
    float left_avg         = avg_window(left_idx,        side_window_deg);
    float right_avg        = avg_window(right_idx,       side_window_deg);


    // Decide and publish -- Function call --
    computeDirection(front_avg, front_left_avg, front_right_avg, left_avg, right_avg);

    ROS_INFO("SIZE: %d", size);
    ROS_INFO("avg range on front: %.3f", front_avg);
    ROS_INFO("avg range on front-right: %.3f", front_right_avg);
    ROS_INFO("avg range on front-left: %.3f", front_left_avg);
    ROS_INFO("Iteration #%d", iteration_count);

}



int main(int argc, char **argv)
{

	ros::init(argc, argv, "laser_avoidance");
	ros::NodeHandle nh("~");  // "~" means use private namespace, so params are scoped to this node

	// Load parameters from the Parameter Server (populated by YAML via launch file).
	// Second argument = variable to store in. Third argument = default if param not found.
	nh.param("linear_speed",          linear_speed,          0.3);
	nh.param("turn_speed",            turn_speed,            0.3);
	nh.param("hard_stop_distance",    hard_stop_distance,    0.95);
	nh.param("min_distance_threshold", min_distance_threshold, 0.1);
	nh.param("direct_front_deg",      direct_front_deg,      20.5);
	nh.param("front_window_deg",      front_window_deg,      23.5);
	nh.param("side_window_deg",       side_window_deg,       35.0);

	ROS_INFO("Parameters loaded: linear=%.2f, turn=%.2f, hard_stop=%.2f",
	         linear_speed, turn_speed, hard_stop_distance);

	pub = nh.advertise<geometry_msgs::Twist>("/husky_velocity_controller/cmd_vel", 100);
	ros::Subscriber sub = nh.subscribe("/scan", 10, laserCallback);

	ros::spin();
	return 0;
}
