import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Given points
points = [
    (1, 2),
    (10, 3),
    (4, 8),
    (15, 7),
    (6, 1),
    (9, 12),
    (14, 4),
    (3, 10),
    (7, 7),
    (12, 14)
]

# Target point
target = (20, 20)

# 1. Calculate distances between (20,20) and each point
print("1. Calculate distances between (20,20) and each point:")
distances = []
for i, point in enumerate(points, 1):
    dist = np.sqrt((target[0] - point[0])**2 + (target[1] - point[1])**2)
    distances.append(dist)
    print(f"Point {i}{point} distance to (20,20): {dist:.2f}")

# 2. Find a point that minimizes the maximum distance to all points

# Define objective function: calculate the maximum distance to all points
def max_distance(point, points):
    max_dist = 0
    for p in points:
        dist = np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2)
        if dist > max_dist:
            max_dist = dist
    return max_dist

# Use scipy's minimize function to solve
initial_guess = np.mean(points, axis=0)  # Use the average of all points as initial guess

result = minimize(
    lambda x: max_distance(x, points),
    initial_guess,
    method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
)

minimax_point = result.x
min_max_distance = result.fun

print("\n2. Find the point that minimizes the maximum distance:")
print(f"Optimal point coordinates: ({minimax_point[0]:.4f}, {minimax_point[1]:.4f})")
print(f"Minimized maximum distance: {min_max_distance:.4f}")

# Verify results: calculate distances from optimal point to each point
print("\nDistances from optimal point to each point:")
for i, point in enumerate(points, 1):
    dist = np.sqrt((minimax_point[0] - point[0])**2 + (minimax_point[1] - point[1])**2)
    print(f"Distance to point {i}{point}: {dist:.4f}")

# Visualize results
plt.figure(figsize=(10, 8))

# Plot all given points
for i, point in enumerate(points, 1):
    plt.scatter(point[0], point[1], color='blue', s=100)
    plt.annotate(f"{i}", (point[0], point[1]), xytext=(5, 5), textcoords='offset points')

# Plot target point (20,20)
plt.scatter(target[0], target[1], color='red', s=100, label='Target point (20,20)')

# Plot optimal point
plt.scatter(minimax_point[0], minimax_point[1], color='green', marker='*', s=200, label='Optimal point')

# Draw the minimum enclosing circle
circle = plt.Circle((minimax_point[0], minimax_point[1]), min_max_distance, color='green', fill=False, linestyle='--')
plt.gca().add_patch(circle)

# Draw lines from optimal point to each point
for point in points:
    plt.plot([minimax_point[0], point[0]], [minimax_point[1], point[1]], 'k--', alpha=0.3)

plt.grid(True)
plt.legend()
plt.title('Point that Minimizes Maximum Distance')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.axis('equal')
plt.tight_layout()
plt.show()