#!/usr/bin/env python3
import math
from typing import Optional
from collections import deque

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


def yaw_to_quat(yaw: float):
    """Yaw (rad) -> quaternion (x,y,z,w)."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def apply_deadzone(v: float, dz: float) -> float:
    return 0.0 if abs(v) < dz else v


class JoyIntegratePoseNode(Node):
    """
    Joy axes -> (vx, vy, wz) velocity command
    integrate -> PoseStamped
    additionally publishes recent trajectory as nav_msgs/Path
    """

    def __init__(self):
        super().__init__("joy_integrate_pose_node")

        # ---- params ----
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("pose_topic", "sim/pose")
        self.declare_parameter("path_topic", "sim/path")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 50.0)

        # Path behavior
        self.declare_parameter("path_duration_sec", 5.0)      # keep last N seconds
        self.declare_parameter("path_publish_rate_hz", 10.0)  # path publish rate (independent)

        # Axis mapping (Joy.axes index)
        self.declare_parameter("axis_vx", 1)
        self.declare_parameter("axis_vy", 0)
        self.declare_parameter("axis_wz", 2)

        # Scaling (units: m/s and rad/s)
        self.declare_parameter("scale_vx", 0.8)
        self.declare_parameter("scale_vy", 0.8)
        self.declare_parameter("scale_wz", 1.5)

        self.declare_parameter("deadzone", 0.05)

        # Integration behavior
        self.declare_parameter("cmd_in_body_frame", True)
        self.declare_parameter("z_height", 0.0)
        self.declare_parameter("wrap_yaw", True)

        # Initial pose
        self.declare_parameter("x0", 0.0)
        self.declare_parameter("y0", 0.0)
        self.declare_parameter("yaw0", 0.0)

        # ---- state ----
        self.x = float(self.get_parameter("x0").value)
        self.y = float(self.get_parameter("y0").value)
        self.yaw = float(self.get_parameter("yaw0").value)

        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.wz_cmd = 0.0

        self.last_time: Optional[rclpy.time.Time] = None

        # Path buffer: (t_sec, PoseStamped)
        self.path_buf: deque[tuple[float, PoseStamped]] = deque()

        # ---- I/O ----
        joy_topic = str(self.get_parameter("joy_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        path_topic = str(self.get_parameter("path_topic").value)

        self.sub = self.create_subscription(Joy, joy_topic, self.on_joy, 10)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.path_pub = self.create_publisher(Path, path_topic, 10)

        # timers
        pose_rate = float(self.get_parameter("publish_rate_hz").value)
        pose_period = 1.0 / max(pose_rate, 1e-3)
        self.timer = self.create_timer(pose_period, self.on_timer)

        path_rate = float(self.get_parameter("path_publish_rate_hz").value)
        path_period = 1.0 / max(path_rate, 1e-3)
        self.path_timer = self.create_timer(path_period, self.on_path_timer)

        self.get_logger().info(
            f"Started. sub={joy_topic} pose_pub={pose_topic} path_pub={path_topic} frame_id={self.get_parameter('frame_id').value}"
        )

    def on_joy(self, msg: Joy):
        dz = float(self.get_parameter("deadzone").value)

        axis_vx = int(self.get_parameter("axis_vx").value)
        axis_vy = int(self.get_parameter("axis_vy").value)
        axis_wz = int(self.get_parameter("axis_wz").value)

        scale_vx = float(self.get_parameter("scale_vx").value)
        scale_vy = float(self.get_parameter("scale_vy").value)
        scale_wz = float(self.get_parameter("scale_wz").value)

        def get_axis(i: int) -> float:
            if i < 0 or i >= len(msg.axes):
                return 0.0
            return float(msg.axes[i])

        raw_vx = -apply_deadzone(get_axis(axis_vx), dz)
        raw_vy = -apply_deadzone(get_axis(axis_vy), dz)
        raw_wz = -apply_deadzone(get_axis(axis_wz), dz)

        self.vx_cmd = raw_vx * scale_vx
        self.vy_cmd = raw_vy * scale_vy
        self.wz_cmd = raw_wz * scale_wz

    def on_timer(self):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            self.publish_pose(now)        # publish once
            self.update_path(now)         # push to buffer
            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # integrate yaw
        self.yaw += self.wz_cmd * dt
        if bool(self.get_parameter("wrap_yaw").value):
            self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi

        vx = self.vx_cmd
        vy = self.vy_cmd
        cmd_in_body = bool(self.get_parameter("cmd_in_body_frame").value)

        if cmd_in_body:
            cy = math.cos(self.yaw)
            sy = math.sin(self.yaw)
            vx_map = cy * vx - sy * vy
            vy_map = sy * vx + cy * vy
        else:
            vx_map = vx
            vy_map = vy

        self.x += vx_map * dt
        self.y += vy_map * dt

        self.publish_pose(now)
        self.update_path(now)

    def publish_pose(self, stamp: rclpy.time.Time):
        msg = PoseStamped()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)

        msg.pose.position.x = float(self.x)
        msg.pose.position.y = float(self.y)
        msg.pose.position.z = float(self.get_parameter("z_height").value)

        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.pose_pub.publish(msg)

    def update_path(self, stamp: rclpy.time.Time):
        """
        keep last path_duration_sec seconds of PoseStamped in buffer
        """
        t_sec = stamp.nanoseconds * 1e-9
        duration = float(self.get_parameter("path_duration_sec").value)
        if duration <= 0.0:
            self.path_buf.clear()
            return

        # store current pose as PoseStamped
        ps = PoseStamped()
        ps.header.stamp = stamp.to_msg()
        ps.header.frame_id = str(self.get_parameter("frame_id").value)

        ps.pose.position.x = float(self.x)
        ps.pose.position.y = float(self.y)
        ps.pose.position.z = float(self.get_parameter("z_height").value)

        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        self.path_buf.append((t_sec, ps))

        # drop old
        t_min = t_sec - duration
        while self.path_buf and self.path_buf[0][0] < t_min:
            self.path_buf.popleft()

    def on_path_timer(self):
        """
        publish Path at a lower rate (e.g., 10Hz) to reduce overhead
        """
        if not self.path_buf:
            return

        now = self.get_clock().now()
        msg = Path()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)
        msg.poses = [ps for (_, ps) in self.path_buf]
        self.path_pub.publish(msg)


def main():
    rclpy.init()
    node = JoyIntegratePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
