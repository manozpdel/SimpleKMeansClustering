import numpy as np  # Importing numpy for numerical operations
import random  # Importing random for random sampling

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        # Initializes the KMeans algorithm with default values for number of clusters and maximum iterations
        self.n_clusters = n_clusters  # Number of clusters to form
        self.max_iter = max_iter  # Maximum number of iterations to perform
        self.controls = None  # Initialize controls as None

    def fit_predict(self, X):
        # Fits the KMeans model to the data and returns the cluster assignments for each data point
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)  # Randomly select initial centroids
        self.centroids = X[random_index]  # Initialize centroids with random samples

        for i in range(self.max_iter):  # Iterate until convergence or maximum iterations reached
            cluster_group = self.assign_clusters(X)  # Assign data points to clusters
            old_centroids = self.centroids  # Store old centroids
            self.controls = self.move_centroids(X, cluster_group)  # Move centroids
            if (old_centroids == self.centroids).all():  # Check for convergence
                break
        return cluster_group  # Return cluster assignments

    def assign_clusters(self, X):
        # Assigns each data point to the nearest centroid
        cluster_group = []  # Initialize list to store cluster assignments
        distances = []  # Initialize list to store distances

        for row in X:  # Iterate over each data point
            for centroid in self.centroids:  # Iterate over each centroid
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))  # Calculate distance
            min_distance = min(distances)  # Find minimum distance
            index_pos = distances.index(min_distance)  # Find index of nearest centroid
            cluster_group.append(index_pos)  # Assign data point to nearest cluster
            distances.clear()  # Clear distance list for next data point

        return np.array(cluster_group)  # Return cluster assignments as numpy array

    def move_centroids(self, X, cluster_group):
        # Moves the centroids to the mean of the data points assigned to each cluster
        new_centroids = []  # Initialize list to store new centroids
        cluster_type = np.unique(cluster_group)  # Find unique cluster indices

        for type in cluster_type:  # Iterate over each cluster
            new_centroids.append(X[cluster_group == type].mean(axis=0))  # Calculate mean of data points in cluster

        return new_centroids  # Return updated centroids
