import time
import numpy as np
import matplotlib.pyplot as plt
from lidar_simulator.map_data import MAP_DATA
from lidar_simulator.sim_utilities import solve_ray_line_segment_intersection

# @title generate point cloud data from lidar

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

        for line_points in MAP_DATA:
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

def closest_point_on_segment(
    point: np.ndarray,      # [x, y]
    line_segment: np.ndarray  # [[x1, y1], [x2, y2]]
) -> tuple[np.ndarray, float]:
    A = line_segment[0]
    B = line_segment[1]

    # Calculate vector from A to B
    AB_vec = B - A

    # Calculate vector from A to the input point
    AP_vec = point - A

    # Compute the projection parameter t
    # Handle case where segment length is zero (A and B are the same point)
    segment_length_squared = np.dot(AB_vec, AB_vec)
    if segment_length_squared == 0:
        # Segment is a single point, A is the closest point
        closest_point = A
    else:
        t = np.dot(AP_vec, AB_vec) / segment_length_squared
        # Clamp t to be between 0 and 1
        t_clamped = np.clip(t, 0, 1)

        # Calculate the closest point on the segment
        closest_point = A + t_clamped * AB_vec

    # Calculate the Euclidean distance between the input point and the closest point
    distance = np.linalg.norm(point - closest_point)

    return closest_point, distance

if __name__ == "__main__":
    # Example usage:
    point_to_check = np.array([0.7, 0.7])
    segment = np.array([[0, 0], [1, 0]])
    closest, dist = closest_point_on_segment(point_to_check, segment)

    print(f"Point: {point_to_check}")
    print(f"Segment: {segment}")
    print(f"Closest point on segment: {closest}")
    print(f"Distance: {dist}")

    # Another example for visualization
    point_to_check_2 = np.array([0.5, 0.8])
    segment_2 = np.array([[0.2, 0.2], [0.8, 0.2]])
    closest_2, dist_2 = closest_point_on_segment(point_to_check_2, segment_2)

    plt.figure(figsize=(6, 6))
    plt.plot(segment_2[:, 0], segment_2[:, 1], 'b-', linewidth=2, label='Line Segment')
    plt.scatter(point_to_check_2[0], point_to_check_2[1], color='red', marker='o', s=100, label='Query Point')
    plt.scatter(closest_2[0], closest_2[1], color='green', marker='x', s=100, label='Closest Point on Segment')
    plt.plot([point_to_check_2[0], closest_2[0]], [point_to_check_2[1], closest_2[1]], 'k--')
    plt.title('Closest Point on Line Segment Example')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def estimate_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 1. Calculate centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # 2. Center both sets of points
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # 3. Compute the covariance matrix H
    H = source_centered.T @ target_centered

    # 4. Perform SVD on H
    U, S, Vt = np.linalg.svd(H)

    # 5. Calculate the rotation matrix R
    R = Vt.T @ U.T

    # Ensure it's a pure rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt_modified = np.copy(Vt)
        Vt_modified[-1, :] *= -1  # Flip the sign of the last row in Vt
        R = Vt_modified.T @ U.T

    # 6. Calculate the translation vector t
    t = target_centroid - R @ source_centroid

    return R, t

def generate_random_correspondences(
    point_cloud: np.ndarray, # N x 2 array of points from lidar scan
    map_data: np.ndarray    # M x 2 x 2 array of line segments representing the map
) -> tuple[np.ndarray, np.ndarray]:
    # Filter out NaN points from the point cloud
    valid_indices = ~np.isnan(point_cloud).any(axis=1)
    valid_point_cloud = point_cloud[valid_indices]

    if len(valid_point_cloud) < 3:
        raise ValueError("Not enough valid points in the point cloud to select 3 distinct points.")

    # Randomly select three distinct points from the valid point cloud
    # Using replace=False to ensure distinct points
    selected_indices = np.random.choice(len(valid_point_cloud), 3, replace=False)
    source_points = valid_point_cloud[selected_indices]

    target_points = []

    # For each selected source point, find its closest corresponding point on the map
    for s_point in source_points:
        min_dist = float('inf')
        closest_map_point_for_s = None

        for line_segment in map_data:
            current_closest_point, current_distance = closest_point_on_segment(s_point, line_segment)
            if current_distance < min_dist:
                min_dist = current_distance
                closest_map_point_for_s = current_closest_point

        if closest_map_point_for_s is not None:
            target_points.append(closest_map_point_for_s)
        else:
            # This case should ideally not happen if map_data is not empty and valid
            # For robustness, we can add a placeholder or raise an error
            # For now, let's append a NaN array
            target_points.append(np.array([np.nan, np.nan]))

    return np.array(source_points), np.array(target_points)

if __name__ == "__main__":
    try:
        source_correspondences, target_correspondences = generate_random_correspondences(point_cloud, MAP_DATA)

        print("Generated 3-point correspondences:")
        for i in range(3):
            print(f"Source Point {i+1}: {source_correspondences[i]} -> Target Point {i+1}: {target_correspondences[i]}")

        # Optional: Visualization of the correspondences
        plt.figure(figsize=(6, 6))
        for line in MAP_DATA:
            plt.plot(line[:, 0], line[:, 1], color="gray", linestyle='--', alpha=0.7)
        plt.scatter(point_cloud[:, 0], point_cloud[:, 1], color="red", s=5, label="Original Point Cloud")
        plt.scatter(source_correspondences[:, 0], source_correspondences[:, 1], color="blue", s=50, marker='o', label="Selected Source Points")
        plt.scatter(target_correspondences[:, 0], target_correspondences[:, 1], color="green", s=50, marker='x', label="Corresponding Target Points")

        for i in range(3):
            plt.plot([source_correspondences[i, 0], target_correspondences[i, 0]],
                    [source_correspondences[i, 1], target_correspondences[i, 1]],
                    'k--', alpha=0.5)

        plt.title('3-Point Correspondences for RANSAC-ICP')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    except ValueError as e:
        print(f"Error generating correspondences: {e}")

def build_segments(map_data):
    """
    map_data: list of segments or polylines
      - segment: (2,2)
      - polyline: (M,2), M>=2 -> (M-1) segments
    return:
      A: (S,2), AB: (S,2), inv_len2: (S,)
    """
    segA = []
    segB = []
    for item in map_data:
        p = np.asarray(item, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
            continue
        a = p[:-1]
        b = p[1:]
        segA.append(a)
        segB.append(b)

    if not segA:
        return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)))

    A = np.vstack(segA)
    B = np.vstack(segB)
    AB = B - A
    len2 = np.sum(AB * AB, axis=1)
    inv_len2 = 1.0 / np.maximum(len2, 1e-12)
    return A, AB, inv_len2

def count_inliers_fast(
    valid_point_cloud, A, AB, inv_len2, R, t, residual_threshold, chunk=2048
):
    """
    valid_point_cloud: (N,2) NaN除去済み
    A, AB, inv_len2: build_segments() の出力
    R: (2,2), t: (2,)
    """
    if len(valid_point_cloud) == 0 or len(A) == 0:
        return 0

    squared_thr = residual_threshold * residual_threshold

    # (R @ P.T).T より (P @ R.T) の方が速いことが多い
    P = valid_point_cloud @ R.T + t  # (N,2) 点群を推定されたR,tで変換

    cnt = 0  # インライアの数をカウントするためのカウンタを初期化
    S = A.shape[0]  # マップ内の線分(segments)の総数

    # 点群をチャンク（塊）に分けて処理することでメモリ効率と速度を向上
    for i in range(0, len(P), chunk):
        Pc = P[i:i+chunk]                       # (C,2) 現在のチャンクの点群 (Cはチャンクサイズ)

        AP = Pc[:, None, :] - A[None, :, :]     # (C,S,2) 各点と各線分の始点間のベクトル

        u = np.sum(AP * AB[None, :, :], axis=2) * inv_len2[None, :] # (C,S) 各点に対する各線分上のパラメータ
        u = np.clip(u, 0.0, 1.0) # パラメータuを0から1の間にクランプし、射影点が線分外に出ないようにする

        proj = A[None, :, :] + u[:, :, None] * AB[None, :, :]  # (C,S,2) 各点から各線分上の射影点

        # d = Pc - proj: 各点と射影点との間のベクトル
        d = Pc[:, None, :] - proj # (C,S,2) 各点と各線分上の射影点間のベクトル
        dist2 = np.sum(d * d, axis=2)          # (C,S) 各点から各線分への距離の二乗
        min_dist2 = np.min(dist2, axis=1)      # (C,) 各点から最も近い線分への最小距離の二乗

        cnt += int(np.count_nonzero(min_dist2 <= squared_thr)) # インライア数を加算

    return cnt

def ransac_icp(
    point_cloud: np.ndarray, # Original point cloud
    map_data: np.ndarray,    # Map data (list of line segments)
    max_iterations: int,
    residual_threshold: float
) -> tuple[np.ndarray, np.ndarray, int, dict]: # Added dict for timings
    best_inlier_count = 0
    best_R = np.eye(2)
    best_t = np.zeros(2)

    # Initialize timing aggregators
    total_correspondence_time = 0.0
    total_transform_time = 0.0
    total_inlier_time = 0.0

    valid_pc = point_cloud[~np.isnan(point_cloud).any(axis=1)]
    A, AB, inv_len2 = build_segments(map_data)

    for _ in range(max_iterations):
        try:
            # a. Call generate_random_correspondences to get source_correspondences and target_correspondences.
            start_correspondence = time.time()
            source_correspondences, target_correspondences = generate_random_correspondences(point_cloud, map_data)
            end_correspondence = time.time()
            total_correspondence_time += (end_correspondence - start_correspondence)

            # b. Call estimate_rigid_transform with these correspondences to get a candidate rotation matrix R_candidate and translation vector t_candidate.
            start_transform = time.time()
            R_candidate, t_candidate = estimate_rigid_transform(source_correspondences, target_correspondences)
            end_transform = time.time()
            total_transform_time += (end_transform - start_transform)

            # c. Call count_inliers with the original point_cloud, map_data, R_candidate, t_candidate, and residual_threshold to get current_inlier_count.
            start_inlier = time.time()
            current_inlier_count = count_inliers_fast(
                valid_pc, A, AB, inv_len2,
                R_candidate, t_candidate,
                residual_threshold
            )
            end_inlier = time.time()
            total_inlier_time += (end_inlier - start_inlier)

            # d. If current_inlier_count is greater than best_inlier_count, update best_inlier_count, best_R, and best_t.
            if current_inlier_count > best_inlier_count:
                best_inlier_count = current_inlier_count
                best_R = R_candidate
                best_t = t_candidate
        except ValueError as e:
            # Handle cases where generate_random_correspondences might fail (e.g., not enough valid points)
            # or other numerical issues in transformation estimation
            # print(f"Skipping iteration due to error: {e}") # Optionally print for debugging
            continue

    # After the loop, return best_R, best_t, and best_inlier_count, and detailed timings.
    timings = {
        "generate_random_correspondences": total_correspondence_time,
        "estimate_rigid_transform": total_transform_time,
        "count_inliers": total_inlier_time
    }
    return best_R, best_t, best_inlier_count, timings

if __name__ == "__main__":
    # Call the ransac_icp function with appropriate parameters
    MAX_ITERATIONS = 2000 # Increased iterations for better chance of finding a good model
    RESIDUAL_THRESHOLD = 0.05 # Reusing the previous test threshold

    best_R_ransac, best_t_ransac, best_inlier_count_ransac, detailed_timings = ransac_icp(point_cloud, MAP_DATA, MAX_ITERATIONS, RESIDUAL_THRESHOLD)

    # Print the best_R, best_t, and best_inlier_count found by the RANSAC algorithm to evaluate the initial results.
    print(f"\n--- RANSAC ICP Results ---")
    print(f"Best Inlier Count: {best_inlier_count_ransac}")
    print(f"Best Rotation Matrix R:\n{best_R_ransac}")
    print(f"Best Translation Vector t:\n{best_t_ransac}")
    print(f"\nDetailed RANSAC internal timings (total over {MAX_ITERATIONS} iterations):\n")
    for step, total_time in detailed_timings.items():
        print(f"  {step}: {total_time:.4f} seconds")

    # Optional: Visualize the final transformed point cloud with the best RANSAC transformation
    valid_point_cloud_final = point_cloud[~np.isnan(point_cloud).any(axis=1)]
    transformed_pc_final = (best_R_ransac @ valid_point_cloud_final.T).T + best_t_ransac

    plt.figure(figsize=(7, 7))
    for line in MAP_DATA:
        plt.plot(line[:, 0], line[:, 1], color="gray", linestyle='--', alpha=0.7)
    plt.scatter(valid_point_cloud_final[:, 0], valid_point_cloud_final[:, 1], color="blue", s=5, alpha=0.3, label="Original Point Cloud")
    plt.scatter(transformed_pc_final[:, 0], transformed_pc_final[:, 1], color="green", s=10, label="Transformed Point Cloud (RANSAC Best)")
    plt.title('RANSAC ICP: Final Transformed Point Cloud')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()