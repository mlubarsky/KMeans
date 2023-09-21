import numpy as np
import random

class cluster:
    def __init__(self, n_clusters=5):
        self.k = n_clusters
        self.centroids = None

    # place the initial centroids
    def initialize_centroids(self, X):
        # select the first centroid
        centroids = [X[random.randint(0, len(X) - 1)]]
        while len(centroids) < self.k:
            # calculate the squared distance from each point to the nearest centroid
            distances = np.array([min(np.linalg.norm(c - x) ** 2 for c in centroids) for x in X])
            # choose the next centroid with the probability distribution based on distances, and add it to the centroids np array
            prob = distances / np.sum(distances)
            next_centroid = X[np.random.choice(len(X), p=prob)]
            centroids.append(next_centroid)
        return np.array(centroids)
        
    # helper function to find nearest centroid
    def nearest_centroid(self, x, centroids):
        # calculate distance of data points from all the centroids
        distances = np.linalg.norm(centroids - x, axis=1)
        # return the index of closest centroid
        return np.argmin(distances)

    # helepr function to move around centroids
    def move_centroid(self, X, labels, centroid_idx):
        cluster_points = X[labels == centroid_idx]
        # check for data points in the cluster
        if len(cluster_points) == 0:
            return self.centroids[centroid_idx]
        # calculate new centroid using the mean of the data points
        new_centroid = np.mean(cluster_points, axis=0)
        return new_centroid

    # fit kmeans on the data
    def fit(self, X, max_iter=100):
        # set centroid
        self.centroids = self.initialize_centroids(X) 
        labels = np.zeros(X.shape[0])
        for iteration in range(max_iter):
            # assign each row to a centroid
            for i in range(X.shape[0]):
                labels[i] = self.nearest_centroid(X[i], self.centroids)
            # move centroids
            for c in range(self.k):
                self.centroids[c] = self.move_centroid(X, labels, c)
        # return cluster assignments
        return labels

    # evaluates data points based on centroid location
    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            # find distances
            distances = np.linalg.norm(self.centroids - x, axis=1) ** 2
            # find indices of centroids
            centroid_idx = np.argmin(distances)
            # add centroids and their indices to arrays
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs   
