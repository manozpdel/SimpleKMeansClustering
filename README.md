# KMeans Clustering Algorithm Implementation

This repository contains a Python implementation of the KMeans clustering algorithm from scratch.

## Overview

KMeans is a popular unsupervised machine learning algorithm used for clustering similar data points into groups or clusters. This implementation provides a basic version of the KMeans algorithm, including centroid initialization, cluster assignment, and centroid updating.

## Features

- Initialization of centroids with random samples.
- Iterative assignment of data points to the nearest centroid.
- Update of centroids based on the mean of data points assigned to each cluster.
- Convergence check to stop iterations when centroids no longer change.

## Usage

To use the KMeans algorithm, follow these steps:

1. Import the `KMeans` class from the `kmeans.py` module.
2. Create an instance of the `KMeans` class with desired parameters such as the number of clusters (`n_clusters`) and maximum iterations (`max_iter`).
3. Fit the KMeans model to your data using the `fit_predict` method, passing the data as input.
4. Retrieve the cluster assignments for each data point.

Example usage:
```python
from kmeans import KMeans
import numpy as np

# Generate sample data
data = np.random.rand(100, 2)

# Create KMeans instance
kmeans = KMeans(n_clusters=3, max_iter=100)

# Fit the model and get cluster assignments
cluster_assignments = kmeans.fit_predict(data)

print(cluster_assignments)
```

## Requirements

- Python 3.x
- NumPy

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
