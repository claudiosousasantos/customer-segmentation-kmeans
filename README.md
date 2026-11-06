# Customer Segmentation with K-Means

A Python script that groups telecom customers into segments based on how they use their service, using a K-means clustering algorithm written from scratch with NumPy (no machine learning library).

## How it works
- Takes usage data for 12 customers: data (GB), voice (minutes), and SMS (count)
- **Normalizes** each column (subtracts the mean, divides by the standard deviation) so a feature measured in hundreds of minutes doesn't outweigh one measured in single-digit GB
- Runs K-means with 3 clusters:
  1. Calculates each customer's distance to each centroid
  2. Assigns each customer to the nearest centroid
  3. Moves each centroid to the average of its customers
  4. Repeats until the centroids stop moving (or 10 iterations)
- Reports each segment's members and average usage in real units

## Example results (sample data)
| Segment | Customers | Avg data | Avg voice | Avg SMS |
|---------|-----------|----------|-----------|---------|
| Data Streamers | C01-C04 | 46.3 GB | 32.5 min | 22.5 |
| Talkers | C05-C08 | 5.8 GB | 395.0 min | 42.5 |
| Texters | C09-C12 | 2.0 GB | 22.5 min | 165.0 |

## How to run
```bash
python kmeans_segmentation.py
```

## What I learned
- Why features on different scales must be normalized before measuring distances
- Using broadcasting (`X[:, None, :] - centroids[None, :, :]`) to calculate every customer-to-centroid distance at once, with no nested loops
- Using `.argmin(axis=1)` to assign each point to its nearest centroid
- Using boolean masks (`X[labels == c]`) to select the members of a cluster
- Stopping an iterative algorithm when it converges, with a maximum iteration count as a safety net
- Converting results back to real units so they can be interpreted

## Limitations
- The sample data is built to have three obvious groups, so the clustering is easy. Real data is much messier.
- The starting centroids are picked by hand (one customer from each group), which makes the result predictable. Real implementations use random starts or k-means++.
- The number of clusters (3) is chosen up front. Real projects compare several values of k.

## Dependencies
Requires NumPy:
```bash
pip install numpy
```
