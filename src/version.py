import torch, torchvision
print(torchvision.__version__)         # PyTorch version
print(torch.version.cuda)        # CUDA version used by PyTorch
print(torch.cuda.is_available()) # Check if CUDA is available

import numpy as np
import matplotlib.pyplot as plt

# Fix B as in the example
B = np.array([[2.0, 0.0],
              [0.0, 0.5]])

def f_slice(x1, y1):
    """
    Restrict x=(x1,0), y=(y1,0).
    Then f(x,y) = || x y^T - B ||_F^2.
    """
    # Construct the rank-1 matrix x y^T
    M = np.array([[x1 * y1, 0.0],
                  [0.0,      0.0]])
    return np.sum((M - B)**2)  # Frobenius norm squared

# Create a grid of x1,y1 values for plotting
n = 200
x1_vals = np.linspace(-4, 4, n)
y1_vals = np.linspace(-4, 4, n)

# Compute f on this 2D slice
F = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        F[i, j] = f_slice(x1_vals[i], y1_vals[j])

# Convert x1_vals, y1_vals to a 2D mesh for contour plotting
X1, Y1 = np.meshgrid(x1_vals, y1_vals)

# Plot a contour map
plt.figure(figsize=(7,6))
contours = plt.contourf(X1, Y1, F, levels=50, cmap='viridis')
plt.colorbar(contours)
plt.xlabel('$x_1$')
plt.ylabel('$y_1$')
plt.title(r'Contour plot of $f(x_1,y_1) = \|\,x y^T - B\|_F^2$ on slice $(x_1,0),(y_1,0)$')
plt.show()
