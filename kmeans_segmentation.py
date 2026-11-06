import numpy as np

customer_ids = np.array(["C01", "C02", "C03", "C04", "C05", "C06",
                         "C07", "C08", "C09", "C10", "C11", "C12"])
usage = np.array([
    [45,  30,  20], [50,  25,  15], [42,  40,  30], [48,  35,  25],
    [ 5, 400,  40], [ 8, 350,  50], [ 4, 450,  35], [ 6, 380,  45],
    [ 2,  20, 150], [ 3,  15, 180], [ 1,  30, 160], [ 2,  25, 170],
])

# Normalize each column
mean = usage.mean(axis=0)
std = usage.std(axis=0)
X = (usage - mean) / std

# K-means
k = 3
centroids = X[[0, 4, 8]].copy()

for iteration in range(10):
    diff = X[:, None, :] - centroids[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    labels = distances.argmin(axis=1)

    new_centroids = np.zeros_like(centroids)
    for c in range(k):
        new_centroids[c] = X[labels == c].mean(axis=0)

    moved = np.abs(new_centroids - centroids).max()
    centroids = new_centroids
    if moved < 1e-6:
        break

# Describe segments in real units
segment_names = {0: "Data Streamers", 1: "Talkers", 2: "Texters"}

for c in range(k):
    members = customer_ids[labels == c]
    avg = usage[labels == c].mean(axis=0)
    print(f"Segment {c} ({segment_names[c]}): {len(members)} customers {members}")
    print(f"   avg data: {avg[0]:.1f} GB | voice: {avg[1]:.1f} min | SMS: {avg[2]:.1f}")