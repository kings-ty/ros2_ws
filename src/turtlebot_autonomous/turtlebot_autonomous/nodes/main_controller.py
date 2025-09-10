import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import cv2
import numpy as np

from ..models.yolo_detector import YoloDetector
from ..config.parameters import RobotConfig
from ..core.state_machine import StateMachine, RobotState
from ..utils.image_processor import ImageProcessor
from ..utils.laser_processor import LaserProcessor
from ..models.dqn_model import DQNAgent
from ..core.reward_calculator import RewardCalculator

class AutonomousController(Node):
    """Enhanced autonomous controller with YOLO detection and intelligent navigation"""
    
    def __init__(self):
        super().__init__('autonomous_controller')
        
        # Initialize components
        self.config = RobotConfig()
        self.state_machine = StateMachine(self.get_logger())
        self.image_processor = ImageProcessor()
        self.laser_processor = LaserProcessor()
        self.reward_calculator = RewardCalculator(self.config)
        
        # Training mode (set to True to enable RL training)
        self.training_mode = False
        self.episode_steps = 0
        self.episode_reward = 0.0

        # ROS2 interfaces
        self.setup_ros_interfaces()
        
        # AI components
        self.setup_ai_components()
        
        # State variables
        self.laser_data = None
        self.current_image = None
        self.detections = []
        self.target_object = None
        self.obstacles = []
        
        # Performance tracking
        self.frame_count = 0
        self.last_action = 0
        self.stuck_counter = 0
        
        # Skid steer specific variables
        self.previous_linear_vel = 0.0
        self.previous_angular_vel = 0.0
        self.velocity_history = []
        self.terrain_adaptation_mode = False
        self.last_successful_maneuver = 1  # Default forward action
        
        self.get_logger().info('Enhanced Autonomous Controller initialized for TurtleBot4!')
        self.get_logger().info(f'4WD Skid Steer Mode: ENABLED')
        self.get_logger().info(f'YOLO available: {self.yolo_detector.is_available()}')
        self.get_logger().info(f'Training mode: {self.training_mode}')
        self.get_logger().info(f'Velocity smoothing: {self.config.VELOCITY_SMOOTHING}')
    
    def setup_ros_interfaces(self):
        """Setup ROS2 publishers and subscribers"""
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        # Camera re-enabled for 4WD rover with added camera sensor
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        print("SETUP_ROS_INTERFACE")
    
    def setup_ai_components(self):
        """Setup AI/ML components"""
        # Initialize YOLO detector
        self.yolo_detector = YoloDetector(confidence_threshold=0.5)
        
        # Initialize DQN agent for training (optional)
        if self.training_mode:
            self.dqn_agent = DQNAgent(
                laser_size=360,
                image_size=(3, 60, 80),
                action_size=len(self.config.ACTIONS),
                lr=0.001,
                epsilon=1.0
            )
            self.get_logger().info('DQN agent initialized for training')
    
    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg
    
    def image_callback(self, msg):
        """Process camera images with YOLO detection"""
        self.current_image = self.image_processor.ros_to_cv2(msg)
        self.frame_count += 1
        
        # Process YOLO detections every 3rd frame for performance
        if self.current_image is not None and self.frame_count % 3 == 0:
            if self.yolo_detector.is_available():
                try:
                    # Run YOLO detection
                    self.detections, self.target_object, self.obstacles = self.yolo_detector.detect_objects(
                        self.current_image,
                        target_classes=self.config.TARGET_CLASSES,
                        avoid_classes=['chair', 'couch', 'table', 'car', 'truck', 'bicycle']
                    )
                    
                    # Log detection results
                    if self.target_object:
                        self.get_logger().info(
                            f'Target detected: {self.target_object["class"]} '
                            f'at ({self.target_object["center_x"]:.0f}, {self.target_object["center_y"]:.0f})',
                            throttle_duration_sec=2.0
                        )
                    
                except Exception as e:
                    self.get_logger().error(f'YOLO detection error: {e}')
                    self.detections = []
                    self.target_object = None
                    self.obstacles = []
            else:
                self.detections = []
                self.target_object = None
                self.obstacles = []
    
    def control_loop(self):
        """Main control loop - enhanced with intelligent decision making"""
        if not self.is_ready():
            print("Break times")
            return
        
        # Update state machine
        self.update_state_machine()
        
        # Get action based on current state
        if self.training_mode and hasattr(self, 'dqn_agent'):
            # Use DQN agent for action selection
            action = self.dqn_agent.act(self.laser_data, self.current_image, training=True)
        else:
            # Use rule-based action selection
            action = self.get_action()
        
        # Execute action
        self.execute_action(action)
        
        # Training logic
        if self.training_mode and hasattr(self, 'dqn_agent'):
            self.handle_training_step(action)
        
        # Update visualization
        self.update_display()
        
        # Update performance tracking
        self.update_performance_tracking(action)
        
        # Check for terrain adaptation (skid steer specific)
        if self._detect_terrain_issues():
            self.get_logger().warn('Terrain adaptation mode activated - adjusting for traction issues')
    
    def is_ready(self):
        """Check if all required sensors are available"""
        laser_ready = self.laser_data is not None
        
        if not laser_ready:
            self.get_logger().warn('Laser data not available', throttle_duration_sec=5.0)
        
        # Only require laser data for 4WD rover
        return laser_ready
    
    def update_state_machine(self):
        """Update robot state based on sensor data and detections"""
        # Analyze laser data with improved filtering
        min_front, min_left, min_right, min_front_narrow = self.laser_processor.analyze_sectors(self.laser_data)
        
        # Enhanced logging for debugging distance calculations
        self.get_logger().info(
            f'Distances - Front: {min_front:.3f}m, Narrow: {min_front_narrow:.3f}m, '
            f'Left: {min_left:.3f}m, Right: {min_right:.3f}m',
            throttle_duration_sec=1.0
        )
        
        if min_front_narrow is None:
            return  # No valid laser data
        
        # Additional validation for distance readings
        # Ensure readings are within reasonable bounds for the robot's environment
        min_front_narrow = self._validate_distance(min_front_narrow)
        min_front = self._validate_distance(min_front)
        
        # Enhanced state transition logic with hysteresis to prevent oscillation
        current_state = self.state_machine.get_current_state()
        
        # Emergency state - immediate obstacle very close
        if min_front_narrow < self.config.EMERGENCY_DISTANCE:
            self.state_machine.transition_to(RobotState.EMERGENCY)
            self.get_logger().warn(
                f'EMERGENCY: Front distance {min_front_narrow:.3f}m < {self.config.EMERGENCY_DISTANCE}m'
            )
        
        # Following state - target detected and path is clear
        elif self.has_target() and min_front > self.config.SAFE_DISTANCE + 0.1:  # Add hysteresis
            self.state_machine.transition_to(RobotState.FOLLOWING)
        
        # Avoiding state - obstacle detected within safe distance
        elif min_front < self.config.SAFE_DISTANCE:
            # Only transition to avoiding if not already in emergency
            if current_state != RobotState.EMERGENCY:
                self.state_machine.transition_to(RobotState.AVOIDING)
                self.get_logger().info(
                    f'AVOIDING: Front distance {min_front:.3f}m < {self.config.SAFE_DISTANCE}m'
                )
        
        # Searching state - no immediate obstacles, exploring
        else:
            # Add small hysteresis to prevent rapid state changes
            if current_state == RobotState.AVOIDING and min_front < self.config.SAFE_DISTANCE + 0.15:
                pass  # Stay in avoiding state until clearly safe
            else:
                self.state_machine.transition_to(RobotState.SEARCHING)
    
    def _validate_distance(self, distance):
        """
        Validate and clean distance readings
        """
        if distance is None:
            return self.laser_data.range_max if self.laser_data else 10.0
        
        # Clamp distance to reasonable bounds
        min_valid = 0.05  # 5cm minimum (sensor limitation)
        max_valid = 10.0  # 10m maximum (typical indoor range)
        
        return max(min_valid, min(distance, max_valid))
    
    def _smooth_velocity(self, target_vel, previous_vel, alpha):
        """
        Apply exponential smoothing to velocity commands for better traction
        """
        return alpha * previous_vel + (1.0 - alpha) * target_vel
    
    def _apply_acceleration_limits(self, target_linear, previous_linear):
        """
        Apply acceleration limits to prevent wheel slipping in skid steer
        """
        dt = 0.1  # Control loop period (10Hz)
        max_accel = self.config.ACCELERATION_LIMIT
        max_decel = self.config.DECELERATION_LIMIT
        
        vel_diff = target_linear - previous_linear
        max_vel_change_accel = max_accel * dt
        max_vel_change_decel = max_decel * dt
        
        if vel_diff > 0:  # Accelerating
            limited_diff = min(vel_diff, max_vel_change_accel)
        else:  # Decelerating
            limited_diff = max(vel_diff, -max_vel_change_decel)
        
        return previous_linear + limited_diff
    
    def _detect_terrain_issues(self):
        """
        Detect if robot is having traction issues and adapt behavior
        """
        # Keep track of velocity history
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)
        
        current_vel = abs(self.previous_linear_vel) + abs(self.previous_angular_vel)
        self.velocity_history.append(current_vel)
        
        # If consistently low velocity despite commands, might be stuck on terrain
        if len(self.velocity_history) >= 5:
            avg_vel = sum(self.velocity_history) / len(self.velocity_history)
            if avg_vel < 0.05 and self.last_action != 0:  # Should be moving but aren't
                self.terrain_adaptation_mode = True
                return True
        
        return False
    
    def get_action(self):
        """Get action based on current state using rule-based logic"""
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
        """Search behavior optimized for skid steer - explore environment looking for targets"""
        min_front, min_left, min_right, _ = self.laser_processor.analyze_sectors(self.laser_data)
        
        # Skid steer can handle more aggressive exploration
        if self.stuck_counter > 15:  # Stuck detection for 4WD
            # Use reverse maneuvers available with skid steer
            if self.stuck_counter > 30:
                return 9 if min_left > min_right else 10  # Reverse + turn
            else:
                return 4 if min_left > min_right else 5   # Pivot turn
        
        # Enhanced search behavior leveraging 4WD capabilities
        if min_front > 2.0:  # Plenty of space - use higher speed
            return 1  # Fast forward
        elif min_front > 1.2:  # Good space - moderate speed
            # Gentle exploration turns
            if min_left > min_right + 0.5:
                return 7  # Slow forward + left
            elif min_right > min_left + 0.5:
                return 8  # Slow forward + right
            else:
                return 1  # Forward
        elif min_front > 0.8:  # Limited space - careful movement
            return 7 if min_left > min_right else 8  # Slow forward + turn
        else:  # Tight space - use skid steer pivot capability
            return 4 if min_left > min_right else 5   # Pivot turn
    
    def follow_action(self):
        """Following behavior - track and follow detected target"""
        if not self.target_object or not self.current_image:
            return 1  # Default forward if no target info
        
        # Calculate target position relative to image center
        image_center_x = self.current_image.shape[1] / 2
        target_x = self.target_object['center_x']
        x_error = target_x - image_center_x
        x_error_normalized = x_error / image_center_x
        
        # Estimate distance based on object size (rough approximation)
        estimated_distance = max(0.5, 8000 / self.target_object['area'])
        
        # Decision logic for following
        if abs(x_error_normalized) < 0.15:  # Target is well centered
            if estimated_distance > 2.5:
                return 1  # Move forward - target too far
            elif estimated_distance < 1.0:
                return 6  # Move backward - target too close
            else:
                return 0  # Stop - good distance
        elif x_error_normalized > 0.3:
            return 3  # Forward + Right - target is significantly right
        elif x_error_normalized < -0.3:
            return 2  # Forward + Left - target is significantly left
        elif x_error_normalized > 0:
            return 5  # Turn right - target slightly right
        else:
            return 4  # Turn left - target slightly left
    
    def avoid_action(self):
        """Enhanced obstacle avoidance for skid steer - navigate around obstacles"""
        min_front, min_left, min_right, _ = self.laser_processor.analyze_sectors(self.laser_data)
        
        # Skid steer can make tighter turns and handle more complex maneuvers
        space_difference = abs(min_left - min_right)
        
        if min_front < 0.4:  # Very close obstacle - use pivot turn
            return 4 if min_left > min_right else 5  # Pivot turn
        elif space_difference > 0.4:  # Significant space difference
            if min_left > min_right:
                return 7  # Slow forward + left (smooth arc)
            else:
                return 8  # Slow forward + right (smooth arc)
        elif space_difference > 0.2:  # Moderate space difference
            if min_left > min_right:
                return 2  # Forward + left (tighter arc)
            else:
                return 3  # Forward + right (tighter arc)
        else:  # Similar space on both sides - choose based on recent success
            # Use adaptive behavior - remember what worked before
            if hasattr(self, 'last_successful_maneuver'):
                if self.last_successful_maneuver in [2, 4, 7]:  # Left turns were successful
                    return 7  # Try slow left
                else:
                    return 8  # Try slow right
            else:
                return 7 if min_left > min_right else 8
    
    def emergency_action(self):
        """Enhanced emergency behavior for skid steer - back away from immediate danger"""
        min_front, min_left, min_right, _ = self.laser_processor.analyze_sectors(self.laser_data)
        
        # Skid steer can perform complex emergency maneuvers
        if min_front < 0.15:  # Extremely close - immediate reverse
            return 6  # Straight backward
        elif min_left > min_right + 0.2:  # More space on left
            return 9   # Reverse + left turn
        elif min_right > min_left + 0.2:  # More space on right
            return 10  # Reverse + right turn
        else:  # Equal space - straight back then decide
            return 6   # Backward
    
    def execute_action(self, action):
        """Execute the selected action with skid steer optimizations"""
        if not (0 <= action < len(self.config.ACTIONS)):
            action = 0  # Default to stop for invalid actions
        
        linear_vel, angular_vel = self.config.ACTIONS[action]
        
        # Apply velocity smoothing for better traction and stability
        if self.config.VELOCITY_SMOOTHING:
            linear_vel = self._smooth_velocity(linear_vel, self.previous_linear_vel, self.config.VELOCITY_ALPHA)
            angular_vel = self._smooth_velocity(angular_vel, self.previous_angular_vel, self.config.VELOCITY_ALPHA)
        
        # Apply acceleration limits to prevent wheel slipping
        linear_vel = self._apply_acceleration_limits(linear_vel, self.previous_linear_vel)
        
        # Store previous velocities for next iteration
        self.previous_linear_vel = linear_vel
        self.previous_angular_vel = angular_vel
        
        # Track successful maneuvers for adaptive behavior
        if action != 0:  # Not stopping
            self.last_successful_maneuver = action
        
        # Create and publish twist message
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        
        self.cmd_vel_pub.publish(twist)
        
        # Log action periodically - updated for skid steer actions
        action_names = [
            'STOP', 'FORWARD', 'FWD_LEFT', 'FWD_RIGHT', 'TURN_LEFT', 'TURN_RIGHT', 
            'BACKWARD', 'SLOW_FWD_LEFT', 'SLOW_FWD_RIGHT', 'REV_LEFT', 'REV_RIGHT'
        ]
        if action < len(action_names):
            action_name = action_names[action]
        else:
            action_name = f'ACTION_{action}'
        
        current_state = self.state_machine.get_current_state().value
        self.get_logger().info(
            f'State: {current_state} | Action: {action_name} | '
            f'Target: {"YES" if self.target_object else "NO"}',
            throttle_duration_sec=1.0
        )
    
    def has_target(self):
        """Check if any target objects are detected"""
        return self.target_object is not None
    
    def update_display(self):
        """Update visual display with detections and status"""
        # Skip display if no camera available
        if self.current_image is None:
            return
        
        # Create display image with detections
        if self.yolo_detector.is_available() and self.detections:
            display_image = self.yolo_detector.draw_detections(
                self.current_image, self.detections, self.target_object)
        else:
            display_image = self.current_image.copy()
        
        # Add status overlay
        y_offset = 30
        
        # Robot state
        state_text = f"State: {self.state_machine.get_current_state().value}"
        cv2.putText(display_image, state_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Detection count
        detection_text = f"Detections: {len(self.detections)}"
        cv2.putText(display_image, detection_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        # Target information
        if self.target_object:
            target_text = f"Target: {self.target_object['class']} ({self.target_object['confidence']:.2f})"
            cv2.putText(display_image, target_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Laser information
        if self.laser_data:
            min_front, _, _, _ = self.laser_processor.analyze_sectors(self.laser_data)
            if min_front is not None:
                laser_text = f"Min Distance: {min_front:.2f}m"
                cv2.putText(display_image, laser_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Skid steer specific information
        y_offset += 25
        skid_steer_text = f"4WD Mode: Smoothing={self.config.VELOCITY_SMOOTHING}"
        cv2.putText(display_image, skid_steer_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 20
        
        # Terrain adaptation status
        if hasattr(self, 'terrain_adaptation_mode') and self.terrain_adaptation_mode:
            terrain_text = "Terrain Adaptation: ACTIVE"
            cv2.putText(display_image, terrain_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += 20
        
        # Velocity information
        vel_text = f"Vel: L={self.previous_linear_vel:.2f} A={self.previous_angular_vel:.2f}"
        cv2.putText(display_image, vel_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Training information
        if self.training_mode and hasattr(self, 'dqn_agent'):
            training_text = f"Training: Eps={self.dqn_agent.epsilon:.3f} Steps={self.episode_steps}"
            cv2.putText(display_image, training_text, (10, display_image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Show the image
        cv2.imshow('Enhanced Autonomous Navigation', display_image)
        cv2.waitKey(1)
    
    def update_performance_tracking(self, action):
        """Track performance and detect stuck conditions"""
        # Track if robot is stuck (same action repeatedly without progress)
        if action == self.last_action and action in [0, 4, 5]:  # Stop or turning actions
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        self.last_action = action
        
        # Log if stuck
        if self.stuck_counter > 30:
            self.get_logger().warn(f'Robot appears stuck! Counter: {self.stuck_counter}')
    
    def handle_training_step(self, action):
        """Handle reinforcement learning training step"""
        if not hasattr(self, 'dqn_agent'):
            return
        
        # Calculate reward
        reward, reward_breakdown = self.reward_calculator.calculate_reward(
            self.laser_data, action, self.detections, self.target_object
        )
        
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Store experience and train
        # Note: This is simplified - full implementation would need previous state storage
        if len(self.dqn_agent.memory) > self.dqn_agent.batch_size:
            loss = self.dqn_agent.replay()
            
            # Log training progress
            if self.episode_steps % 50 == 0:
                self.get_logger().info(
                    f'Training Step {self.episode_steps}: '
                    f'Reward={reward:.2f} Total={self.episode_reward:.2f} '
                    f'Loss={loss:.4f} Epsilon={self.dqn_agent.epsilon:.3f}'
                )


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    controller = AutonomousController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Enhanced Autonomous Controller...')
    
    # Cleanup
    cv2.destroyAllWindows()
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()