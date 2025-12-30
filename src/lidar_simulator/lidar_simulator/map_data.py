# @title generate and visualize map data

import matplotlib.pyplot as plt
import numpy as np

MAP_DATA = np.array([
    [[0, 0], [0, 8]],
    [[0, 8], [8, 8]],
    [[8, 8], [8, 0]],
    [[8, 0], [0, 0]],
    [[0, 5], [5, 5]],
    [[2, 2], [8, 2]],
])

if __name__ == '__main__':
    plt.figure(figsize=(5, 5))
    for line in MAP_DATA:
        plt.plot(line[:, 0], line[:, 1])
    plt.show()