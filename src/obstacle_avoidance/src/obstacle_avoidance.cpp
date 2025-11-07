#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <string>

//Change these constants for vehicle kinematics
#define DEFAULT_LINEAR 0.5	     //LINEAR SPEED for no obstacle detected
#define DEFAULT_ANGULAR 0	      //ANGULAR SPEED (Turn)
#define DISTANCE 0.775		       //Maximum distance to consider point an obstacle
#define NEW_LINEARX 0.3	       //Velocity
#define TURN_ANGULAR_SPEED 0.3	   //Turn speed
#define STUCK_ANGULAR_SPEED 0.45
#define MINIMUM_DISTANCE_THRESHOLD 0.1 //how sensitive LiDAR is to small distance values (DEFAULT: 0.1)
#define DIRECT_FRONT_DEG 18.5
#define FRONT_WINDOW_DEG 19.5            // averaging window around front (± degrees)
#define SIDE_WINDOW_DEG 35             // averaging window for side look


ros::Publisher pub;

void computeDirection(float front_avg, float front_left_avg, float front_right_avg,
                      float left_avg,  float right_avg)
{
    geometry_msgs::Twist cmd;
    float linearx = 0.0f, angularz = 0.0f;
    std::string case_desc;
    const float HARD_STOP_DIST = 0.95;
    

    // --- 1. Emergency hard stop if anything close in the forward arc ---
    if (front_avg < HARD_STOP_DIST || front_left_avg < HARD_STOP_DIST || front_right_avg < HARD_STOP_DIST)
    {
        if(front_avg > HARD_STOP_DIST && abs(front_left_avg-front_right_avg) < .1){
            case_desc = "Case: Hard Stop -->  -- FRONT OPEN SIDES EQUAL ---";
            linearx = NEW_LINEARX;
            angularz = 0.0;
        } else {
            case_desc = "Case: Hard Stop —> --- FRONT BLOCKED AND SIDES UNEQUAL---";
            linearx = 0.0;
            angularz = (front_left_avg > front_right_avg) ? +TURN_ANGULAR_SPEED : -TURN_ANGULAR_SPEED;
        }
        
    }
    else
    {
        case_desc = "Case: Clear ahead -> ---MOVE FORWARD---";
        linearx = NEW_LINEARX;
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

    static int iteration_count = 0;   //Iteration Counter
    iteration_count++;


    auto valid = [&](int idx) {
        if (idx < 0 || idx >= size) return false; 
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

    // int front_idx = size / 2;                     
    // assumes angle_min ~ -pi, angle_max ~ +pi
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
    float front_avg        = avg_window(front_idx,       DIRECT_FRONT_DEG);
    float front_left_avg   = avg_window(front_left_idx,  FRONT_WINDOW_DEG);
    float front_right_avg  = avg_window(front_right_idx, FRONT_WINDOW_DEG);
    float left_avg         = avg_window(left_idx,        SIDE_WINDOW_DEG);
    float right_avg        = avg_window(right_idx,       SIDE_WINDOW_DEG);
    

    // Decide and publish
    computeDirection(front_avg, front_left_avg, front_right_avg, left_avg, right_avg);

    ROS_INFO("SIZE: %d", size);
    ROS_INFO("avg range on front: %.3f", front_avg);
    ROS_INFO("avg range on front-right: %.3f", front_right_avg);
    ROS_INFO("avg range on front-left: %.3f", front_left_avg);
    //ROS_INFO("avg range on left:  %.3f", left_avg);
    //ROS_INFO("avg range on right: %.3f", right_avg);
    ROS_INFO("Iteration #%d", iteration_count);
    
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
