import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def lorenz_attractor(x, y, z, s=10, r=28, b=2.667):
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 10000

xs = np.empty((num_steps + 1,))
ys = np.empty((num_steps + 1,))
zs = np.empty((num_steps + 1,))

xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)

for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz_attractor(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs, ys, zs, lw=0.5)
ax.set_title("Lorenz Attractor")

buffer = BytesIO()
plt.savefig(buffer, format="png")
plt.close(fig)
buffer.seek(0)
base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")