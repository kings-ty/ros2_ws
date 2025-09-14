#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class TrainingManager(Node):
    def __init__(self):
        super().__init__('training_manager')
        self.get_logger().info("Training Manager node started (placeholder)")

def main(args=None):
    rclpy.init(args=args)
    node = TrainingManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()