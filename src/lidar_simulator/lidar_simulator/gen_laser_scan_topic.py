from sensor_msgs.msg import LaserScan
import numpy as np

def ranges_to_laserscan(
    ranges: np.ndarray,
    stamp,
    frame_id: str,
    angle_min: float,
    angle_max: float,
    range_min: float,
    range_max: float,
    scan_time: float = 0.0,
) -> LaserScan:
    msg = LaserScan()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    msg.angle_min = float(angle_min)
    msg.angle_max = float(angle_max)
    msg.angle_increment = (angle_max - angle_min) / max(len(ranges) - 1, 1)

    msg.range_min = float(range_min)
    msg.range_max = float(range_max)

    msg.scan_time = float(scan_time)
    msg.time_increment = float(scan_time / max(len(ranges) - 1, 1)) if scan_time > 0 else 0.0

    # NaN→inf, 範囲外→inf（またはクリップ）
    r = np.array(ranges, dtype=np.float32)
    r[~np.isfinite(r)] = np.inf
    r[(r < range_min) | (r > range_max)] = np.inf

    msg.ranges = r.tolist()
    return msg
