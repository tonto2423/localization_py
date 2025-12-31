#!/usr/bin/env python3
import math
import time
from typing import Optional
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path

from tf2_ros import TransformBroadcaster

from lidar_simulator.map_data import MAP_DATA
from ransac_icp.ransac_icp import ransac_icp


def yaw_to_quat(yaw: float):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_to_R(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


class RansacIcpLocalizerNode(Node):
    def __init__(self):
        super().__init__("ransac_icp_localizer")

        # ---- params (I/O) ----
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("pose_topic", "/icp_pose")
        self.declare_parameter("path_topic", "/icp_path")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("child_frame", "laser")

        # ---- params (RANSAC-ICP) ----
        self.declare_parameter("max_iterations", 20)
        self.declare_parameter("residual_threshold", 0.03)

        # ---- params (initial guess) ----
        self.declare_parameter("init_x", 0.8)
        self.declare_parameter("init_y", 0.5)
        self.declare_parameter("init_yaw", 0.0)

        # ---- params (speed) ----
        self.declare_parameter("scan_stride", 1)
        self.declare_parameter("log_every_n", 10)

        # ---- params (Path) ----
        self.declare_parameter("path_duration_sec", 5.0)       # keep last N sec
        self.declare_parameter("path_publish_rate_hz", 20.0)   # publish Path at lower rate

        # ---- params (TF) ----
        self.declare_parameter("publish_tf", True)

        scan_topic = str(self.get_parameter("scan_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        path_topic = str(self.get_parameter("path_topic").value)

        self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)
        self.pub_pose = self.create_publisher(PoseStamped, pose_topic, 10)
        self.pub_path = self.create_publisher(Path, path_topic, 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # map is fixed
        self.map_data = MAP_DATA

        # ---- state: current initial guess (accumulated pose) ----
        init_x = float(self.get_parameter("init_x").value)
        init_y = float(self.get_parameter("init_y").value)
        init_yaw = float(self.get_parameter("init_yaw").value)

        self.current_R = yaw_to_R(init_yaw)
        self.current_t = np.array([init_x, init_y], dtype=np.float64)

        self.frame_count = 0

        # ---- Path buffer ----
        # store (t_sec, PoseStamped)
        self.path_buf: deque[tuple[float, PoseStamped]] = deque()

        # path publish timer
        path_rate = float(self.get_parameter("path_publish_rate_hz").value)
        path_period = 1.0 / max(path_rate, 1e-3)
        self.path_timer = self.create_timer(path_period, self.on_path_timer)

        self.get_logger().info(
            f"Started. sub={scan_topic} pose_pub={pose_topic} path_pub={path_topic} "
            f"max_iter={self.get_parameter('max_iterations').value} "
            f"resid={self.get_parameter('residual_threshold').value}"
        )

    def scan_to_points_lidar(self, msg: LaserScan) -> np.ndarray:
        """LaserScan -> (N,2) points in lidar frame"""
        stride = int(self.get_parameter("scan_stride").value)
        stride = max(1, stride)

        ranges = np.asarray(msg.ranges, dtype=np.float64)
        N = ranges.shape[0]
        if N == 0:
            return np.zeros((0, 2), dtype=np.float64)

        idx = np.arange(0, N, stride, dtype=np.int32)
        ang = msg.angle_min + idx * msg.angle_increment
        r = ranges[idx]

        finite = np.isfinite(r)
        if msg.range_max > 0.0:
            finite &= (r <= msg.range_max)
        if msg.range_min > 0.0:
            finite &= (r >= msg.range_min)

        r = r[finite]
        ang = ang[finite]
        if r.size == 0:
            return np.zeros((0, 2), dtype=np.float64)

        x = r * np.cos(ang)
        y = r * np.sin(ang)
        return np.stack([x, y], axis=1)

    def publish_pose(self, stamp, R: np.ndarray, t: np.ndarray):
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        qx, qy, qz, qw = yaw_to_quat(yaw)

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = str(self.get_parameter("map_frame").value)

        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.pub_pose.publish(msg)
        return msg  # Path用に返す

    def publish_tf_map_to_laser(self, stamp, R: np.ndarray, t: np.ndarray):
        if not bool(self.get_parameter("publish_tf").value):
            return

        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        qx, qy, qz, qw = yaw_to_quat(yaw)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = str(self.get_parameter("map_frame").value)
        tf_msg.child_frame_id = str(self.get_parameter("child_frame").value)

        tf_msg.transform.translation.x = float(t[0])
        tf_msg.transform.translation.y = float(t[1])
        tf_msg.transform.translation.z = 0.0

        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_msg)

    def update_path_buffer(self, pose_msg: PoseStamped):
        # keep last path_duration_sec
        duration = float(self.get_parameter("path_duration_sec").value)
        if duration <= 0.0:
            self.path_buf.clear()
            return

        # PoseStamped.header.stamp is builtin_interfaces/Time
        t_sec = float(pose_msg.header.stamp.sec) + 1e-9 * float(pose_msg.header.stamp.nanosec)

        self.path_buf.append((t_sec, pose_msg))

        t_min = t_sec - duration
        while self.path_buf and self.path_buf[0][0] < t_min:
            self.path_buf.popleft()

    def on_path_timer(self):
        if not self.path_buf:
            return

        now = self.get_clock().now().to_msg()
        msg = Path()
        msg.header.stamp = now
        msg.header.frame_id = str(self.get_parameter("map_frame").value)
        msg.poses = [ps for (_, ps) in self.path_buf]
        self.pub_path.publish(msg)

    def on_scan(self, msg: LaserScan):
        t0 = time.time()

        # 1) scan -> points in lidar frame
        pc_lidar = self.scan_to_points_lidar(msg)
        if pc_lidar.shape[0] == 0:
            pose_msg = self.publish_pose(msg.header.stamp, self.current_R, self.current_t)
            self.publish_tf_map_to_laser(msg.header.stamp, self.current_R, self.current_t)
            self.update_path_buffer(pose_msg)
            return

        # 2) pre-transform using current initial guess: P_map_approx
        pre_pc = (pc_lidar @ self.current_R.T) + self.current_t

        # 3) RANSAC-ICP refine in map frame
        max_it = int(self.get_parameter("max_iterations").value)
        thr = float(self.get_parameter("residual_threshold").value)

        R_refine, t_refine, best_inliers, _tim = ransac_icp(
            pre_pc, self.map_data, max_it, thr
        )

        # 4) combine: new pose (accumulated)
        self.current_R = R_refine @ self.current_R
        self.current_t = (R_refine @ self.current_t) + t_refine

        # 5) publish pose + tf + path update
        pose_msg = self.publish_pose(msg.header.stamp, self.current_R, self.current_t)
        self.publish_tf_map_to_laser(msg.header.stamp, self.current_R, self.current_t)
        self.update_path_buffer(pose_msg)

        # 6) logging
        self.frame_count += 1
        dt = time.time() - t0
        log_every = int(self.get_parameter("log_every_n").value)
        if log_every > 0 and (self.frame_count % log_every == 0):
            yaw_deg = math.degrees(math.atan2(self.current_R[1, 0], self.current_R[0, 0]))
            self.get_logger().info(
                f"frame={self.frame_count}  "
                f"pose=[{self.current_t[0]:.3f},{self.current_t[1]:.3f},{yaw_deg:.1f}deg]  "
                f"inliers={best_inliers}  time={dt*1000:.1f}ms"
            )


def main():
    rclpy.init()
    node = RansacIcpLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
