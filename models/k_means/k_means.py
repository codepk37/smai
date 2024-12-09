import numpy as np

class KMeans:
    def __init__(self, k=5, max_iter=100, tol=0.0001):
        self.k = k #numeber of cluster
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.clusters = None
        self.conver= max_iter

    def fit(self, X):
        # Initialize centroids randomly from the data points
        # random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        # self.centroids = X[random_indices]

        # Initialize centroids using k-means++ algorithm
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign clusters
            self.clusters = self._assign_clusters(X)

            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()

            # Update centroids
            self.centroids = self._update_centroids(X)

            # Check for convergence
            if np.all(np.abs(self.centroids - old_centroids) < self.tol):
                # print(f"Convergence reached at {_}")
                self.conver= _
                break
        
    def predict(self, X):
        return self._assign_clusters(X)

    def getCost(self, X):
        cost = 0
        for i in range(self.k):
            cluster_points = X[self.clusters == i]
            cost += np.sum((cluster_points - self.centroids[i])**2)
        return cost
    
    def _initialize_centroids(self, X):
        # Randomly choose the first centroid
        centroids = [X[np.random.choice(X.shape[0])]]
        
        for _ in range(1, self.k):
            # Compute distances from the nearest centroid for each point ,For each data point, compute the distance to the nearest already-chosen centroid. This gives you a measure of how "far" each point is from the nearest centroid.
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
            
            # Compute probabilities based on squared distances
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            
            # Choose a new centroid randomly based on probabilities
            new_centroid = X[np.random.choice(X.shape[0], p=probabilities)] # you might select a point with the highest probability, but it’s still random
            centroids.append(new_centroid)
        
        return np.array(centroids)


    def _assign_clusters(self, X): #below is explaination
        # Compute the distance between each point and each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)


    """def _assign_clusters(self, X):
        # Initialize an array to store distances between each point and each centroid
        num_points = X.shape[0]
        num_centroids = self.centroids.shape[0]
        distances = np.zeros((num_points, num_centroids))
        
        # Calculate the distance between each point and each centroid
        for i in range(num_points):
            for j in range(num_centroids):
                # Compute Euclidean distance between point i and centroid j
                distances[i, j] = np.sqrt(np.sum((X[i] - self.centroids[j]) ** 2))
        
        # Assign each point to the nearest centroid
        # For each point, find the index of the centroid with the minimum distance
        cluster_assignments = np.argmin(distances, axis=1)
        
        return cluster_assignments
    """

    def _update_centroids(self, X):
        # Compute the mean of points in each cluster to update centroids
        new_centroids = np.array([X[self.clusters == i].mean(axis=0) for i in range(self.k)])
        return new_centroids
