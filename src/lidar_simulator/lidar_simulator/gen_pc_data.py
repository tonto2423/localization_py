# @title gen_ranges (LaserScan-like) + test plot + consistency test

import numpy as np
import matplotlib.pyplot as plt
from .sim_utilities import solve_ray_line_segment_intersection
from .map_data import MAP_DATA

def gen_ranges(
    pose_2d: np.ndarray,
    num_point: int,
    map_data: np.ndarray,
    *,
    angle_min: float = -np.pi,
    angle_max: float = np.pi,
    range_min: float = 0.02,
    range_max: float = 20.0,
    noise_std: float | None = None,
    outlier_probability: float = 0.0,
    outlier_mode: str = "no_return",  # "no_return" | "max_range" | "random_range"
) -> np.ndarray:
    """
    物理レイキャストで ranges (N,) を生成（LaserScan想定）

    - 角度はセンサ基準：theta_sensor in [angle_min, angle_max]
    - レイキャスト方向は世界基準：theta_world = yaw + theta_sensor
    - 未検出は np.inf
    - ノイズは range に加える（一般的な距離ノイズ）
    """
    x0, y0, yaw = float(pose_2d[0]), float(pose_2d[1]), float(pose_2d[2])
    origin = np.array([x0, y0], dtype=np.float64)

    if num_point <= 0:
        return np.zeros((0,), dtype=np.float32)

    # N本の角度（endpoint含む）
    if num_point == 1:
        thetas_sensor = np.array([angle_min], dtype=np.float64)
    else:
        thetas_sensor = np.linspace(angle_min, angle_max, num_point, dtype=np.float64)

    ranges = np.full((num_point,), np.inf, dtype=np.float32)

    for i, th_s in enumerate(thetas_sensor):
        # --- outlier ---
        if outlier_probability > 0.0 and (np.random.rand() < outlier_probability):
            if outlier_mode == "no_return":
                ranges[i] = np.inf
            elif outlier_mode == "max_range":
                # max付近（ほんの少し揺らす）
                r = range_max * (0.95 + 0.05 * np.random.rand())
                ranges[i] = float(np.clip(r, range_min, range_max))
            elif outlier_mode == "random_range":
                r = range_min + (range_max - range_min) * np.random.rand()
                ranges[i] = float(r)
            else:
                raise ValueError(f"unknown outlier_mode: {outlier_mode}")
            continue

        # --- ray casting ---
        th_w = yaw + th_s
        ray_dir = np.array([np.cos(th_w), np.sin(th_w)], dtype=np.float64)

        best = np.inf
        for line_points in map_data:
            hit = solve_ray_line_segment_intersection(origin, ray_dir, line_points)
            if hit is None:
                continue

            r = float(np.linalg.norm(hit - origin))
            if r < best:
                best = r

        # --- measurement noise & limits ---
        if np.isfinite(best):
            if noise_std is not None and noise_std > 0:
                best = best + float(np.random.normal(0.0, noise_std))

            # range clip: 範囲外は「返ってこない」に寄せて inf 扱いにするのが一般的
            if (best < range_min) or (best > range_max) or (best <= 0.0):
                ranges[i] = np.inf
            else:
                ranges[i] = float(best)
        else:
            ranges[i] = np.inf

    return ranges


def ranges_to_points_world(
    pose_2d: np.ndarray,
    ranges: np.ndarray,
    *,
    angle_min: float = -np.pi,
    angle_max: float = np.pi,
) -> np.ndarray:
    """
    表示用：ranges -> 点群（世界座標）
    infはNaN点にする
    """
    x0, y0, yaw = float(pose_2d[0]), float(pose_2d[1]), float(pose_2d[2])
    N = len(ranges)
    pts = np.full((N, 2), np.nan, dtype=np.float64)

    if N == 0:
        return pts

    thetas_sensor = np.array([angle_min], dtype=np.float64) if N == 1 else np.linspace(angle_min, angle_max, N)

    for i, (r, th_s) in enumerate(zip(ranges, thetas_sensor)):
        if not np.isfinite(r):
            continue
        th_w = yaw + float(th_s)
        pts[i, 0] = x0 + float(r) * np.cos(th_w)
        pts[i, 1] = y0 + float(r) * np.sin(th_w)

    return pts


# ---- optional: consistency check against your old implementation (when outliers=0, noise=None) ----
def gen_pc_data_reference(
    pose_2d: np.ndarray,
    num_point: int,
    map_data: np.ndarray,
) -> np.ndarray:
    """
    あなたの gen_pc_data の「rangesだけ」版（基準）
    - 角度: pose_yaw + 2π/N * it
    - 未検出: NaN
    """
    ranges = []
    x0, y0 = pose_2d[:2]

    for it in range(num_point):
        theta_i = pose_2d[2] + 2 * np.pi / num_point * it
        ray_dir = np.array([np.cos(theta_i), np.sin(theta_i)], dtype=np.float64)

        best = np.inf
        for line_points in map_data:
            hit = solve_ray_line_segment_intersection(np.array([x0, y0]), ray_dir, line_points)
            if hit is None:
                continue
            r = float(np.linalg.norm(hit - pose_2d[:2]))
            if r < best:
                best = r

        ranges.append(best if np.isfinite(best) else np.nan)

    return np.array(ranges, dtype=np.float64)


if __name__ == "__main__":
    pose_2d = np.array([0.5, 0.8, np.deg2rad(30)], dtype=np.float64)
    num_point = 50

    # ここを 2π/N に合わせたいなら、LaserScan側もそれに揃える：
    # - referenceは [0, 2π) を等間隔（endpointなし）で回してる
    # - LaserScanは普通 [angle_min, angle_max] を endpoint込みでlinspaceする
    #
    # なので「一致テスト」を成立させたい場合、角度離散を合わせる必要がある。
    # ここでは一致テスト用に reference と同じ離散に合わせるため、angle_min=0, angle_max=2π*(N-1)/N を採用
    angle_min = 0.0
    angle_max = 2.0 * np.pi * (num_point - 1) / num_point  # endpoint込みでちょうど reference の角度列になる

    # --- generate ---
    ranges = gen_ranges(
        pose_2d,
        num_point,
        MAP_DATA,
        angle_min=angle_min,
        angle_max=angle_max,
        range_min=0.02,
        range_max=10.0,
        noise_std=0.001,
        outlier_probability=0.05,
        outlier_mode="max_range",
    )

    # --- visualize ---
    pts = ranges_to_points_world(pose_2d, ranges, angle_min=angle_min, angle_max=angle_max)

    plt.figure(figsize=(5, 5))
    for line in MAP_DATA:
        plt.plot(line[:, 0], line[:, 1], color="gray")

    valid = ~np.isnan(pts).any(axis=1)
    plt.scatter(pts[valid, 0], pts[valid, 1], color="red", s=10)

    arrow_length = 0.1
    plt.arrow(pose_2d[0], pose_2d[1], arrow_length*np.cos(pose_2d[2]), arrow_length*np.sin(pose_2d[2]))
    plt.axis("equal")
    plt.title("gen_ranges() test (points for visualization)")
    plt.show()

    # --- consistency test (noise=None, outlier=0) ---
    ranges_ref = gen_pc_data_reference(pose_2d, num_point, MAP_DATA)
    ranges_new = gen_ranges(
        pose_2d, num_point, MAP_DATA,
        angle_min=angle_min, angle_max=angle_max,
        range_min=0.0, range_max=10.0,
        noise_std=None,
        outlier_probability=0.0,
    )

    # refは未検出 NaN, newは未検出 inf なので揃える
    ref2 = ranges_ref.copy()
    ref2[~np.isfinite(ref2)] = np.inf

    max_abs_err = np.nanmax(np.abs(ref2 - ranges_new))
    print("consistency max_abs_err:", max_abs_err)
