#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "turtlesim/msg/pose.hpp"

using namespace std::chrono_literals;

// State machine for turtle behavior
enum class TurtleState {
    FORWARD,
    TURN
};

class WallAvoider : public rclcpp::Node
{
public:
    WallAvoider() : Node("wall_avoider"), state_(TurtleState::FORWARD)
    {
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 10);
        subscription_ = this->create_subscription<turtlesim::msg::Pose>(
            "turtle1/pose", 10, std::bind(&WallAvoider::pose_callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(
            100ms, std::bind(&WallAvoider::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "Wall avoider node started.");
    }

private:
    void pose_callback(const turtlesim::msg::Pose::SharedPtr msg)
    {
        current_pose_ = *msg;
    }
    
    void timer_callback()
    {
        auto cmd_vel_msg = geometry_msgs::msg::Twist();
        
        // Define wall boundaries (the window size is ~11x11)
        double min_x = 1.0;
        double max_x = 10.0;
        double min_y = 1.0;
        double max_y = 10.0;
        
        // Check if the turtle is close to a wall
        bool near_wall = (current_pose_.x < min_x || current_pose_.x > max_x ||
                         current_pose_.y < min_y || current_pose_.y > max_y);

        switch (state_) {
            case TurtleState::FORWARD:
                if (near_wall) {
                    RCLCPP_WARN(this->get_logger(), "Near wall! Changing state to TURN.");
                    state_ = TurtleState::TURN;
                    // Reset turn timer
                    turn_start_time_ = this->get_clock()->now();
                } else {
                    cmd_vel_msg.linear.x = 2.0; // Move forward
                    cmd_vel_msg.angular.z = 0.0;
                }
                break;
            
            case TurtleState::TURN:
                // Turn until a specified time has passed
                if ((this->get_clock()->now() - turn_start_time_).seconds() > 2.0) {
                    RCLCPP_INFO(this->get_logger(), "Finished turning. Changing state to FORWARD.");
                    state_ = TurtleState::FORWARD;
                    cmd_vel_msg.linear.x = 2.0;
                    cmd_vel_msg.angular.z = 0.0;
                } else {
                    cmd_vel_msg.linear.x = 0.0; // Stop moving forward
                    cmd_vel_msg.angular.z = 1.5; // Turn clockwise
                }
                break;
        }

        publisher_->publish(cmd_vel_msg);
    }
    
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    turtlesim::msg::Pose current_pose_;
    
    TurtleState state_;
    rclcpp::Time turn_start_time_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WallAvoider>());
    rclcpp::shutdown();
    return 0;
}
