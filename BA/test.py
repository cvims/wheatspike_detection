import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints)
plt.savefig('plot.png')  # Save the plot as an image file