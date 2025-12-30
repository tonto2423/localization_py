
import matplotlib.pyplot as plt
import numpy as np

# 2つの直線の交点を求める関数-----------------------------------

"""
2つの直線の交点を求める。直線の方程式はax + by + c = 0とする。
line1: [a1, b1, c1]
line2: [a2, b2, c2]
"""
def solve_2_line_intersection(
    line1: np.ndarray,
    line2: np.ndarray,
) -> np.ndarray:
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    det = a1 * b2 - a2 * b1
    if det == 0:
        return None

    x = -(b2 * c1 - b1 * c2) / det
    y = -(a1 * c2 - a2 * c1) / det

    return np.array([x, y])

if __name__ == "__main__":
    # example
    line1 = np.array([1, 2, 3])
    line2 = np.array([4, 5, 6])
    intersection = solve_2_line_intersection(line1, line2)

    # plot
    x_range_ex1 = np.linspace(0, 10, 100)
    y_line1_ex1 = -(line1[0] * x_range_ex1 + line1[2]) / line1[1]
    y_line2_ex1 = -(line2[0] * x_range_ex1 + line2[2]) / line2[1]
    plt.plot(x_range_ex1, y_line1_ex1, label="line1")
    plt.plot(x_range_ex1, y_line2_ex1, label="line2")
    plt.plot(intersection[0], intersection[1], "ro", label="intersection")
    plt.legend()
    plt.show()

# --------------------------------------------------------------

# ray and line segment intersection --------------------------------

def line_from_2_point(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    a =   p2[1] - p1[1]
    b = -(p2[0] - p1[0])
    c = p2[0] * p1[1] - p1[0] * p2[1]
    return np.array([a, b, c])

def line_from_ray(ray_origin: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
    return line_from_2_point(ray_origin, ray_origin + ray_direction)

# 半直線 (ray) と線分 (line segment) の交点
def solve_ray_line_segment_intersection(
    ray_origin: np.ndarray,     # [x0, y0]
    ray_direction: np.ndarray,  # normalized vector
    line_segment: np.ndarray,   # [[x1, y1], [x2, y2]]
) -> np.ndarray:
    # まず、直線の交点を計算する
    intersection = solve_2_line_intersection(
        line_from_2_point(line_segment[0], line_segment[1]),
        line_from_ray(ray_origin, ray_direction),
    )
    if intersection is None:
        return None

    # rayの向きと一致するかチェックする
    if np.dot(ray_direction, intersection - ray_origin) < 0:
        return None

    # 線分の内部に収まるかチェックする
    if np.dot(line_segment[0] - intersection, line_segment[1] - intersection) > 0:
        return None

    return intersection

if __name__ == "__main__":
    # example 2
    ray_origin = np.array([0.5, 0])
    ray_direction = np.array([1, 1]) / np.linalg.norm(np.array([1, 1]))

    line_segment = np.array([[0, 1], [2, 1]])
    intersection = solve_ray_line_segment_intersection(ray_origin, ray_direction, line_segment)
    print(intersection)

    if intersection is None:
        print("交点なし")
    else:
        print("交点あり")

        plt.figure(figsize=(5, 5))
        # rayの可視化
        plt.arrow(ray_origin[0], ray_origin[1], ray_direction[0], ray_direction[1], head_width=0.1, label="ray")
        # 線分の可視化
        plt.plot(line_segment[:, 0], line_segment[:, 1], color="gray", label="line segment")
        # 交点の可視化
        plt.scatter(intersection[0], intersection[1], color="red", label="intersection")
        plt.legend()
        plt.axis('equal')
        plt.show()

# --------------------------------------------------------------



# --------------------------------------------------------------