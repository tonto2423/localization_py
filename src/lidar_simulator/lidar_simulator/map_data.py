# @title generate and visualize map data

import matplotlib.pyplot as plt
import numpy as np

MAP_DATA = np.array([
    [[0, 0], [0, 1]],
    [[0, 1], [1, 1]],
    [[1, 1], [1, 0]],
    [[1, 0], [0, 0]],
    [[0, 0.5], [0.5, 0.5]],
])

plt.figure(figsize=(5, 5))
for line in MAP_DATA:
    plt.plot(line[:, 0], line[:, 1])
plt.show()