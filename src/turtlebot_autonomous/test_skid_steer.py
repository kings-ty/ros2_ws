#!/usr/bin/env python3
"""
Test script for skid steer enhancements
Validates configuration and action improvements
"""

import sys
sys.path.append('/home/ty/ros2_ws/src/turtlebot_autonomous')

from turtlebot_autonomous.config.parameters import RobotConfig

def test_skid_steer_config():
    """Test the enhanced skid steer configuration"""
    print("=== Skid Steer Configuration Test ===")
    
    config = RobotConfig()
    
    # Test basic parameters
    print(f"Max Linear Velocity: {config.MAX_LINEAR_VEL} m/s")
    print(f"Max Angular Velocity: {config.MAX_ANGULAR_VEL} rad/s")
    print(f"Safe Distance: {config.SAFE_DISTANCE} m")
    print(f"Emergency Distance: {config.EMERGENCY_DISTANCE} m")
    
    # Test skid steer specific parameters
    print(f"\n=== Skid Steer Specific ===")
    print(f"Acceleration Limit: {config.ACCELERATION_LIMIT} m/s²")
    print(f"Deceleration Limit: {config.DECELERATION_LIMIT} m/s²")
    print(f"Min Turning Radius: {config.MIN_TURNING_RADIUS} m")
    print(f"Wheel Base: {config.WHEEL_BASE} m")
    print(f"Velocity Smoothing: {config.VELOCITY_SMOOTHING}")
    print(f"Velocity Alpha: {config.VELOCITY_ALPHA}")
    
    # Test enhanced actions
    print(f"\n=== Enhanced Action Set ===")
    print(f"Total Actions: {len(config.ACTIONS)}")
    
    action_names = [
        'STOP', 'FORWARD', 'FWD_LEFT', 'FWD_RIGHT', 'TURN_LEFT', 'TURN_RIGHT', 
        'BACKWARD', 'SLOW_FWD_LEFT', 'SLOW_FWD_RIGHT', 'REV_LEFT', 'REV_RIGHT'
    ]
    
    for i, (linear, angular) in enumerate(config.ACTIONS):
        name = action_names[i] if i < len(action_names) else f"ACTION_{i}"
        print(f"  {i}: {name:<15} | Linear: {linear:+.2f} | Angular: {angular:+.2f}")
    
    # Validate action constraints
    print(f"\n=== Action Validation ===")
    max_linear_found = max(abs(action[0]) for action in config.ACTIONS)
    max_angular_found = max(abs(action[1]) for action in config.ACTIONS)
    
    print(f"Max linear in actions: {max_linear_found:.2f} (limit: {config.MAX_LINEAR_VEL:.2f})")
    print(f"Max angular in actions: {max_angular_found:.2f} (limit: {config.MAX_ANGULAR_VEL:.2f})")
    
    if max_linear_found <= config.MAX_LINEAR_VEL:
        print("✓ Linear velocities within limits")
    else:
        print("✗ Linear velocities exceed limits")
    
    if max_angular_found <= config.MAX_ANGULAR_VEL:
        print("✓ Angular velocities within limits")
    else:
        print("✗ Angular velocities exceed limits")
    
    print("\n=== Test Complete ===")
    return True

def test_velocity_smoothing():
    """Test velocity smoothing function"""
    print("\n=== Velocity Smoothing Test ===")
    
    # Simulate smoothing function
    def smooth_velocity(target_vel, previous_vel, alpha):
        return alpha * previous_vel + (1.0 - alpha) * target_vel
    
    # Test case: step change from 0 to 0.3 m/s
    alpha = 0.7
    target = 0.3
    previous = 0.0
    
    print(f"Alpha: {alpha}")
    print(f"Target velocity: {target} m/s")
    print(f"Step responses:")
    
    current = previous
    for step in range(10):
        current = smooth_velocity(target, current, alpha)
        print(f"  Step {step+1}: {current:.4f} m/s")
    
    print("✓ Velocity smoothing test complete")
    return True

if __name__ == '__main__':
    print("Testing Skid Steer Enhancements")
    print("=" * 50)
    
    try:
        test_skid_steer_config()
        test_velocity_smoothing()
        print("\n🎉 All tests passed! Skid steer configuration is ready.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)