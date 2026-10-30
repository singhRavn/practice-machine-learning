import numpy as np


def gaussian_kernel(distance, bandwidth):
    return np.exp(-(distance ** 2) / (2 * bandwidth ** 2))


def estimate_bandwidth(X, k=5):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    local_widths = []

    for i in range(n_samples):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists = np.sort(dists)
        dists = dists[1: min(k + 1, len(dists))]
        if len(dists) == 0:
            local_widths.append(1.0)
        else:
            local_widths.append(float(np.median(dists)))

    return float(np.median(local_widths))


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


def mean_shift_adaptive(X, k=5, max_iters=100, tol=1e-4):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    centroids = X.copy()

    for _ in range(max_iters):
        new_centroids = np.zeros_like(centroids)

        for i in range(n_samples):
            distances = np.linalg.norm(X - centroids[i], axis=1)
            dists_sorted = np.sort(distances)
            local_bandwidth = np.median(dists_sorted[1: min(k + 1, len(dists_sorted))])
            local_bandwidth = max(local_bandwidth, 1e-6)
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


def merge_modes(modes, tolerance=1e-2):
    if len(modes) == 0:
        return np.array([])

    merged = [modes[0]]
    for mode in modes[1:]:
        if not any(np.linalg.norm(mode - existing) < tolerance for existing in merged):
            merged.append(mode)

    return np.array(merged)


def assign_to_modes(points, modes):
    points = np.asarray(points, dtype=float)
    modes = np.asarray(modes, dtype=float)
    if len(modes) == 0:
        return np.array([], dtype=int)

    labels = []
    for point in points:
        distances = np.linalg.norm(modes - point, axis=1)
        labels.append(int(np.argmin(distances)))
    return np.array(labels)


def cluster_spread(points, labels, modes):
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    modes = np.asarray(modes, dtype=float)
    spreads = []

    for cluster_id in range(len(modes)):
        cluster_points = points[labels == cluster_id]
        if len(cluster_points) == 0:
            spreads.append(0.0)
        else:
            dists = np.linalg.norm(cluster_points - modes[cluster_id], axis=1)
            spreads.append(float(np.mean(dists)))

    return spreads


X = np.array([
    [1, 2], [1.5, 1.8], [2, 2.2],
    [8, 8], [8.5, 8.1], [9, 8.5],
    [10, 9], [9.5, 10], [10.5, 10.5],
    [3, 3], [3.2, 2.8], [2.8, 3.5],
    [7.8, 8.2], [7.5, 7.9], [8.3, 7.5]
])

global_bandwidth = estimate_bandwidth(X, k=5)

fixed_centers = mean_shift(X, bandwidth=global_bandwidth)
adaptive_centers = mean_shift_adaptive(X, k=5)

fixed_modes = merge_modes(fixed_centers)
adaptive_modes = merge_modes(adaptive_centers)

fixed_labels = assign_to_modes(X, fixed_modes)
adaptive_labels = assign_to_modes(X, adaptive_modes)

fixed_spread = cluster_spread(X, fixed_labels, fixed_modes)
adaptive_spread = cluster_spread(X, adaptive_labels, adaptive_modes)

print("Estimated global bandwidth:", global_bandwidth)
print("\nFixed bandwidth centers:")
print(fixed_centers)
print("Merged fixed modes:")
print(fixed_modes)
print("Fixed labels:")
print(fixed_labels)
print("Fixed cluster spread:")
print(fixed_spread)

print("\nAdaptive bandwidth centers:")
print(adaptive_centers)
print("Merged adaptive modes:")
print(adaptive_modes)
print("Adaptive labels:")
print(adaptive_labels)
print("Adaptive cluster spread:")
print(adaptive_spread)

print("\nSummary:")
print(f"Fixed modes count: {len(fixed_modes)}")
print(f"Adaptive modes count: {len(adaptive_modes)}")
