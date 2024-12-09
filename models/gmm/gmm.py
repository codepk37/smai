

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
import numpy as np

def plot_gmm_memberships(X, memberships):
    # Assign colors to each point based on its highest membership probability (cluster responsibility)
    cluster_assignments = np.argmax(memberships, axis=1)
    
    if X.shape[1] == 2:
        # 2D plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', marker='o', alpha=0.6)
        plt.title('GMM Cluster Assignments (2D)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()
    
    elif X.shape[1] == 3:
        # 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_assignments, cmap='viridis', marker='o', alpha=0.6)
        ax.set_title('GMM Cluster Assignments (3D)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.grid(True)
        plt.show()
    else:
        raise ValueError("Data dimensionality not supported for plotting. Only 2D and 3D data are supported.")


# class GMM:
#     def __init__(self, n_components, max_iter=100, tol=1e-3):
#         self.n_components = n_components
#         self.max_iter = max_iter
#         self.tol = tol
    
#     def _initialize(self, X):
#         n_samples, n_features = X.shape
#         self.weights = np.ones(self.n_components) / self.n_components
#         # Randomly initialize means from data points
#         self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
#         # Initialize each covariance matrix individually
#         self.covariances = np.array([np.cov(X, rowvar=False) + np.eye(n_features) * 1e-6
#                                      for _ in range(self.n_components)])
#         self.resp = np.zeros((n_samples, self.n_components))
    
#     def _e_step(self, X):
#         n_samples, n_features = X.shape
#         log_resp = np.zeros((n_samples, self.n_components))
        
#         for k in range(self.n_components):
#             diff = X - self.means[k]
#             cov_inv = np.linalg.inv(self.covariances[k])
#             term = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
#             log_resp[:, k] = -0.5 * (term + n_features * np.log(2 * np.pi) + 
#                                      np.linalg.slogdet(self.covariances[k])[1])
        
#         log_resp += np.log(self.weights)
#         # Use log-sum-exp trick for numerical stability
#         max_log_resp = np.max(log_resp, axis=1, keepdims=True)
#         log_resp -= max_log_resp
#         resp = np.exp(log_resp)
#         resp /= np.sum(resp, axis=1, keepdims=True)
        
#         self.resp = resp
#         return resp
    
#     def _m_step(self, X):
#         n_samples, n_features = X.shape
#         Nk = np.sum(self.resp, axis=0)
        
#         self.weights = Nk / n_samples
#         self.means = np.dot(self.resp.T, X) / Nk[:, np.newaxis]
        
#         for k in range(self.n_components):
#             diff = X - self.means[k]
#             weighted_diff = self.resp[:, k][:, np.newaxis] * diff
#             self.covariances[k] = np.dot(weighted_diff.T, diff) / Nk[k]
#             # Regularize covariance matrix to avoid singularities
#             self.covariances[k] += np.eye(n_features) * 1e-6
    
#     def _log_likelihood(self, X):
#         n_samples = X.shape[0]
#         log_prob = np.zeros((n_samples, self.n_components))
        
#         for k in range(self.n_components):
#             diff = X - self.means[k]
#             cov_inv = np.linalg.inv(self.covariances[k])
#             term = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
#             log_prob[:, k] = -0.5 * (term + X.shape[1] * np.log(2 * np.pi) + 
#                                      np.linalg.slogdet(self.covariances[k])[1])
        
#         log_prob += np.log(self.weights)
#         # Compute log likelihood with log-sum-exp trick
#         log_likelihood = np.sum(np.log(np.sum(np.exp(log_prob), axis=1) + 1e-10))
#         return log_likelihood
    
#     def fit(self, X):
#         self._initialize(X)
#         prev_log_likelihood = -np.inf
        
#         for i in range(self.max_iter):
#             self.resp = self._e_step(X)
#             self._m_step(X)
#             log_likelihood = self._log_likelihood(X)
            
#             if abs(log_likelihood - prev_log_likelihood) < self.tol:
#                 break
#             prev_log_likelihood = log_likelihood
    
#     def getParams(self):
#         return {
#             'weights': self.weights,
#             'means': self.means,
#             'covariances': self.covariances
#         }
    
#     def getMembership(self):
#         return self.resp
    
#     def getLikelihood(self, X):
#         return self._log_likelihood(X)



# Step 2: Define the GMM class and methods
class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def _initialize(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False) + np.eye(n_features) * 1e-6
                                     for _ in range(self.n_components)])
        self.resp = np.zeros((n_samples, self.n_components))
    
    def _e_step(self, X):
        n_samples, n_features = X.shape
        log_resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov_inv = np.linalg.inv(self.covariances[k])
            term = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
            log_resp[:, k] = -0.5 * (term + n_features * np.log(2 * np.pi) + 
                                     np.linalg.slogdet(self.covariances[k])[1])
        
        log_resp += np.log(self.weights)
        max_log_resp = np.max(log_resp, axis=1, keepdims=True)
        log_resp -= max_log_resp
        resp = np.exp(log_resp)
        resp /= np.sum(resp, axis=1, keepdims=True)
        
        self.resp = resp
        return resp
    
    def _m_step(self, X):
        n_samples, n_features = X.shape
        Nk = np.sum(self.resp, axis=0)
        
        self.weights = Nk / n_samples
        self.means = np.dot(self.resp.T, X) / Nk[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_diff = self.resp[:, k][:, np.newaxis] * diff
            self.covariances[k] = np.dot(weighted_diff.T, diff) / Nk[k]
            self.covariances[k] += np.eye(n_features) * 1e-6
    
    def _log_likelihood(self, X):
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov_inv = np.linalg.inv(self.covariances[k])
            term = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
            log_prob[:, k] = -0.5 * (term + X.shape[1] * np.log(2 * np.pi) + 
                                     np.linalg.slogdet(self.covariances[k])[1])
        
        log_prob += np.log(self.weights)
        log_likelihood = np.sum(np.log(np.sum(np.exp(log_prob), axis=1) + 1e-10))
        return log_likelihood
    
    def fit(self, X):
        self._initialize(X)
        prev_log_likelihood = -np.inf
        
        for i in range(self.max_iter):
            self.resp = self._e_step(X)
            self._m_step(X)
            log_likelihood = self._log_likelihood(X)
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
    
    def getDensity(self, X):
        n_samples, n_features = X.shape
        density = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means[k]
            cov_inv = np.linalg.inv(self.covariances[k])
            term = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
            norm_factor = np.sqrt((2 * np.pi) ** n_features * np.linalg.det(self.covariances[k]))
            density[:, k] = self.weights[k] * np.exp(-0.5 * term) / norm_factor

        return np.sum(density, axis=1)