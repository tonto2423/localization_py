"""
- input:
    - `sim/pose` (geometry_msgs/msg/PoseStamped):
        - frame_id: `map`
- output:
    - `scan` (sensor_msgs/msg/LaserScan):
        - frame_id: `laser`
    - `map_marker` (visualization_msgs/msg/MarkerArray):
        - frame_id: `map`
"""

import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.timer import Timer

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import ColorRGBA

from lidar_simulator.map_data import MAP_DATA
from lidar_simulator.vis_utilities import create_line_marker

class LidarSimNode(Node):
    true_pose_topic: str = 'sim/pose'
    scan_topic: str = 'scan'
    map_marker_topic: str = 'map_marker'
    map_frame_id: str = 'map'
    laser_frame_id: str = 'laser'
    
    true_pose_msg: PoseStamped = None
    def __init__(self):
        super().__init__('lidar_sim_node')
        self.declare_parameter('node.main_loop_rate_ms', 10)
        self.declare_parameter('node.marker_pub_loop_rate_ms', 300)
        
        self.true_pose_sub: Subscription = self.create_subscription(
            PoseStamped,
            self.true_pose_topic,
            self.truePoseCb,
            10)
        self.scan_pub: Publisher = self.create_publisher(
            LaserScan,
            self.scan_topic,
            10)
        self.map_marker_pub: Publisher = self.create_publisher(
            MarkerArray,
            self.map_marker_topic,
            10)
        
        self.main_timer: Timer = self.create_timer(
            self.get_parameter('node.main_loop_rate_ms').value * 1e-3,
            self.mainTimerCb
        )
        self.marker_pub_timer: Timer = self.create_timer(
            self.get_parameter('node.marker_pub_loop_rate_ms').value * 1e-3,
            self.markerPubTimerCb
        )
        
        self.get_logger().info("lidar_sim_node has been initialized")

    def truePoseCb(self, msg: PoseStamped) -> None: self.true_pose_msg = msg

    def mainTimerCb(self) -> None:
        if self.true_pose_msg is None: return
        self.get_logger().info("loop")

    def markerPubTimerCb(self) -> None:
        marker_array = MarkerArray()
        for idx, line_segment in enumerate(MAP_DATA):
            # MAP_DATA contains pairs of 2D points. Use idx as integer marker id,
            # namespace 'map_data' and frame id self.map_frame_id ('map').
            marker = create_line_marker(
                self.get_clock(),
                int(idx),
                'map_data',
                self.map_frame_id,
                line_segment[0],
                line_segment[1],
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8), # Green
            )
            marker_array.markers.append(marker)
        self.map_marker_pub.publish(marker_array)

def main() -> None:
    rclpy.init()
    node = LidarSimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()