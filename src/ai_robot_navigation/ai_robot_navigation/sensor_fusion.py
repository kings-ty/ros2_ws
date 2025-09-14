#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class SensorFusion(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        self.get_logger().info("Sensor Fusion node started (placeholder)")

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusion()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()