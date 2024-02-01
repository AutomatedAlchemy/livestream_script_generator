import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np


def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    c = Z.copy()
    img = np.zeros(Z.shape, dtype=float)
    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask] ** 2 + c[mask]
        img[mask] = i
    return img

# Image settings
width, height = 800, 800
x_min, x_max = -2, 2
y_min, y_max = -2, 2
max_iter = 100

img = mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter)

fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='hot')
ax.axis('off')  # Turn off axis

buffer = BytesIO()
plt.savefig(buffer, format='png')
plt.close(fig)
buffer.seek(0)
base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')