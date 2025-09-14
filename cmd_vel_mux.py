#!/usr/bin/env python3

"""
Command Velocity Multiplexer
- Manages control priority between different systems
- AI DQN has higher priority than Nav2
- Smooth switching between controllers
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.ai_sub = self.create_subscription(Twist, '/cmd_vel_ai', self.ai_callback, 10)
        self.nav2_sub = self.create_subscription(Twist, '/cmd_vel_nav', self.nav2_callback, 10)
        
        # Control state
        self.last_ai_time = 0
        self.last_nav2_time = 0
        self.ai_timeout = 0.5  # If no AI command for 0.5s, switch to Nav2
        self.current_controller = "none"
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_timer)
        
        self.get_logger().info("🚦 Command Velocity Multiplexer started!")
        self.get_logger().info("📡 AI commands: /cmd_vel_ai (HIGH priority)")
        self.get_logger().info("🗺️  Nav2 commands: /cmd_vel_nav (LOW priority)")
        self.get_logger().info("🎯 Output: /cmd_vel")
        
    def ai_callback(self, msg):
        """Handle AI DQN commands (HIGH priority)"""
        self.last_ai_time = time.time()
        self.current_controller = "ai"
        self.cmd_vel_pub.publish(msg)
        
    def nav2_callback(self, msg):
        """Handle Nav2 commands (LOW priority)"""
        self.last_nav2_time = time.time()
        # Only use Nav2 if AI hasn't sent commands recently
        if time.time() - self.last_ai_time > self.ai_timeout:
            self.current_controller = "nav2"
            self.cmd_vel_pub.publish(msg)
    
    def control_timer(self):
        """Monitor control status and handle timeouts"""
        current_time = time.time()
        
        # Check if both controllers are inactive
        ai_active = (current_time - self.last_ai_time) < self.ai_timeout
        nav2_active = (current_time - self.last_nav2_time) < 1.0
        
        if not ai_active and not nav2_active:
            # Stop robot if no controllers are active
            if self.current_controller != "none":
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)
                self.current_controller = "none"
                self.get_logger().info("🛑 No active controllers - stopping robot")
    
    def get_status(self):
        """Get current control status for logging"""
        current_time = time.time()
        ai_age = current_time - self.last_ai_time
        nav2_age = current_time - self.last_nav2_time
        
        return {
            'controller': self.current_controller,
            'ai_age': ai_age,
            'nav2_age': nav2_age
        }

def main(args=None):
    rclpy.init(args=args)
    mux = CmdVelMux()
    
    try:
        rclpy.spin(mux)
    except KeyboardInterrupt:
        # Stop robot on shutdown
        stop_msg = Twist()
        mux.cmd_vel_pub.publish(stop_msg)
        mux.get_logger().info("🛑 Shutting down - robot stopped")
    finally:
        mux.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()