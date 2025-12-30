# @title generate point cloud data from lidar

import numpy as np
import matplotlib.pyplot as plt
from sim_utilities import solve_ray_line_segment_intersection
from map_data import MAP_DATA

"""
- input:
  pose_2d (np.ndarray): pose of lidar in (x, y, θ)
  num_point (int): number of points
  map_data (np.ndarray): map data (list of lines)
  noise_std: (float): standard deviation of noise (default: None)
  outlier_probability: (float): probability of generating an outlier (default: 0.0)
  outlier_magnitude: (float): magnitude of outlier distance from lidar origin (default: 1.0)
- output:
  ranges (np.ndarray): list of ranges
  point_cloud (np.ndarray): list of points
"""
def gen_pc_data(
    pose_2d: np.ndarray,
    num_point: int,
    map_data: np.ndarray,
    noise_std: float = None,
    outlier_probability: float = 0.0,
    outlier_magnitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    ranges = []
    point_cloud = []
    x0, y0 = pose_2d[:2]

    for it in range(num_point):
        # Check if an outlier should be generated for this ray
        if np.random.rand() < outlier_probability:
            # Generate a random outlier point within a certain magnitude around the lidar origin
            random_angle = 2 * np.pi * np.random.rand()
            random_distance = outlier_magnitude * np.random.rand() # Uniformly distributed within magnitude
            outlier_point = np.array([
                x0 + random_distance * np.cos(random_angle),
                y0 + random_distance * np.sin(random_angle)
            ])
            ranges.append(np.linalg.norm(outlier_point - pose_2d[:2]))
            point_cloud.append(outlier_point)
            continue # Skip normal intersection calculation for this ray

        # ビームの直線方程式を求める
        θi = pose_2d[2] + 2 * np.pi / num_point * it
        ray_dir = np.array([np.cos(θi), np.sin(θi)])
        # 各直線との交点を求め、最小のものを使う
        min_distance = float("inf")
        min_point_for_ray = None # Renamed to avoid confusion

        for line_points in map_data:
            intersection = solve_ray_line_segment_intersection(
                np.array([x0, y0]),
                ray_dir,
                line_points,
            )
            if intersection is None: continue

            # Apply noise
            current_intersection_point = intersection
            if noise_std is not None:
                range_noise = np.random.normal(0, noise_std)
                # Apply noise along the ray direction from the intersection point
                current_intersection_point = intersection + ray_dir * range_noise

            distance = np.linalg.norm(current_intersection_point - pose_2d[:2])

            if distance < min_distance:
                min_distance = distance
                min_point_for_ray = current_intersection_point

        # After checking all line segments for the current ray
        if min_point_for_ray is not None:
            ranges.append(min_distance)
            point_cloud.append(min_point_for_ray)
        else:
            # If no intersection was found for this ray, append NaN for range and a NaN point
            ranges.append(np.nan)
            point_cloud.append(np.array([np.nan, np.nan])) # Append a 2-element NaN array for homogeneity

    return np.array(ranges), np.array(point_cloud)

if __name__ == "__main__":
    # シミュレーション
    pose_2d = np.array([0.5, 0.8, np.deg2rad(30)])
    num_point = 50
    ranges, point_cloud = gen_pc_data(pose_2d, num_point, MAP_DATA, noise_std=0.001, outlier_probability=0.05, outlier_magnitude=6.5)

    plt.figure(figsize=(5, 5))
    for line in MAP_DATA:
        plt.plot(line[:, 0], line[:, 1], color="gray")
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], color="red")
    arrow_length = 0.1
    plt.arrow(pose_2d[0], pose_2d[1], arrow_length*np.cos(pose_2d[2]), arrow_length*np.sin(pose_2d[2]))
    plt.axis('equal')
    plt.show()
