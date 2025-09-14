#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')
        self.get_logger().info("Navigation Controller node started (placeholder)")

def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()