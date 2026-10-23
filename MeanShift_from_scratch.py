import numpy as np


def gaussian_kernel(distance, bandwidth):
    return np.exp(-(distance ** 2) / (2 * bandwidth ** 2))


def mean_shift(X, bandwidth=1.0, max_iters=100, tol=1e-4):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    centroids = X.copy()

    for _ in range(max_iters):
        new_centroids = np.zeros_like(centroids)

        for i in range(n_samples):
            distances = np.linalg.norm(X - centroids[i], axis=1)
            weights = gaussian_kernel(distances, bandwidth)
            weights_sum = weights.sum()

            if weights_sum == 0:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = (weights[:, None] * X).sum(axis=0) / weights_sum

        shift = np.linalg.norm(new_centroids - centroids, axis=1)
        centroids = new_centroids

        if np.max(shift) < tol:
            break

    return centroids


def mean_shift_dynamic_bandwidth(X, bandwidth_factor=1.5, max_iters=100, tol=1e-4):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    centroids = X.copy()

    for _ in range(max_iters):
        new_centroids = np.zeros_like(centroids)

        for i in range(n_samples):
            distances = np.linalg.norm(X - centroids[i], axis=1)
            positive_distances = distances[distances > 0]
            local_bandwidth = bandwidth_factor * np.median(positive_distances) if len(positive_distances) > 0 else bandwidth_factor
            weights = gaussian_kernel(distances, local_bandwidth)
            weights_sum = weights.sum()

            if weights_sum == 0:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = (weights[:, None] * X).sum(axis=0) / weights_sum

        shift = np.linalg.norm(new_centroids - centroids, axis=1)
        centroids = new_centroids

        if np.max(shift) < tol:
            break

    return centroids


def unique_modes(centroids, tolerance=1e-2):
    unique = []
    for center in centroids:
        if not any(np.linalg.norm(center - existing) < tolerance for existing in unique):
            unique.append(center)
    return np.array(unique)


X = np.array([
    [1, 2], [1.5, 1.8], [2, 2.2],
    [8, 8], [8.5, 8.1], [9, 8.5],
    [10, 9], [9.5, 10], [10.5, 10.5]
])

centers_fixed = mean_shift(X, bandwidth=1.5)
centers_dynamic = mean_shift_dynamic_bandwidth(X, bandwidth_factor=1.5)

fixed_modes = unique_modes(centers_fixed)
dynamic_modes = unique_modes(centers_dynamic)

print("Fixed bandwidth centers:")
print(centers_fixed)
print("Unique fixed modes:")
print(fixed_modes)

print("\nDynamic bandwidth centers:")
print(centers_dynamic)
print("Unique dynamic modes:")
print(dynamic_modes)

print("\nMode comparison:")
print(f"Fixed: {len(fixed_modes)} modes")
print(f"Dynamic: {len(dynamic_modes)} modes")
