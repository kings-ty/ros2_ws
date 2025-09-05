import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import cv2
import cv2

from ..config.parameters import RobotConfig
from ..core.state_machine import StateMachine, RobotState
from ..utils.image_processor import ImageProcessor
from ..utils.laser_processor import LaserProcessor
from ..models.dqn_model import DQNNetwork
from ..core.reward_calculator import RewardCalculator

class AutonomousController(Node):
    """Main controller node - clean and modular"""
    
    def __init__(self):
        super().__init__('autonomous_controller')
        
        # Initialize components
        self.config = RobotConfig()
        self.state_machine = StateMachine(self.get_logger())
        self.image_processor = ImageProcessor()
        self.laser_processor = LaserProcessor()
        self.reward_calculator = RewardCalculator(self.config)
        
        # ROS2 interfaces
        self.setup_ros_interfaces()
        
        # AI components
        self.setup_ai_components()
        
        # State variables
        self.laser_data = None
        self.current_image = None
        self.detections = []
        
        self.get_logger().info('Autonomous Controller initialized!')
    
    def setup_ros_interfaces(self):
        """Setup ROS2 publishers and subscribers"""
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
    
    def setup_ai_components(self):
        """Setup AI/ML components"""
        # Initialize DQN or other ML models here
        pass
    
    def laser_callback(self, msg):
        self.laser_data = msg
    
    def image_callback(self, msg):
        self.current_image = self.image_processor.ros_to_cv2(msg)
        # Process YOLO detections here
    
    def control_loop(self):
        """Main control loop - clean and readable"""
        if not self.is_ready():
            return
        
        # Update state machine
        self.update_state_machine()
        
        # Get action based on current state
        action = self.get_action()
        
        # Execute action
        self.execute_action(action)
        
        # Update visualization
        self.update_display()
    
    def is_ready(self):
        return self.laser_data is not None and self.current_image is not None
    
    def update_state_machine(self):
        # State transition logic
        current_state = self.state_machine.get_current_state()
        
        # Analyze laser data
        min_front, min_left, min_right, min_front_narrow = self.laser_processor.analyze_sectors(self.laser_data)
        
        # State transitions based on sensor data and detections
        if min_front_narrow < self.config.EMERGENCY_DISTANCE:
            self.state_machine.transition_to(RobotState.EMERGENCY)
        elif self.has_target() and min_front > self.config.SAFE_DISTANCE:
            self.state_machine.transition_to(RobotState.FOLLOWING)
        elif min_front < self.config.SAFE_DISTANCE:
            self.state_machine.transition_to(RobotState.AVOIDING)
        else:
            self.state_machine.transition_to(RobotState.SEARCHING)
    
    def get_action(self):
        """Get action based on current state"""
        state = self.state_machine.get_current_state()
        
        if state == RobotState.SEARCHING:
            return self.search_action()
        elif state == RobotState.FOLLOWING:
            return self.follow_action()
        elif state == RobotState.AVOIDING:
            return self.avoid_action()
        elif state == RobotState.EMERGENCY:
            return self.emergency_action()
        else:
            return 0  # Stop
    
    def search_action(self):
        # Search behavior logic
        return 1  # Forward
    
    def follow_action(self):
        # Following behavior logic
        return 1  # Forward
    
    def avoid_action(self):
        # Avoidance behavior logic
        min_front, min_left, min_right, _ = self.laser_processor.analyze_sectors(self.laser_data)
        return 4 if min_left > min_right else 5  # Turn left or right
    
    def emergency_action(self):
        # Emergency behavior logic
        return 6  # Backward
    
    def execute_action(self, action):
        """Execute the selected action"""
        linear_vel, angular_vel = self.config.ACTIONS[action]
        
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        
        self.cmd_vel_pub.publish(twist)
    
    def has_target(self):
        return any(det['class'] in self.config.TARGET_CLASSES for det in self.detections)
    
    def update_display(self):
        """Update visual display"""
        if self.current_image is not None:
            display_image = self.image_processor.draw_detections(self.current_image, self.detections)
            
            # Add state info
            state_text = f"State: {self.state_machine.get_current_state().value}"
            cv2.putText(display_image, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Autonomous Navigation', display_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    controller = AutonomousController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
