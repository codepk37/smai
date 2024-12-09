import pandas as pd
import numpy as np
class PcaAutoencoder:
    def __init__(self, n_components=None):
        """
        Initialize the PCA Autoencoder.
        
        Parameters:
        n_components (int or None): The number of principal components to retain. If None, all components are retained.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ = None
        self.X_reduced = None

    def fit(self, X):
        """
        Fit the PCA model to the data by calculating eigenvalues and eigenvectors.

        Parameters:
        X (ndarray): The input data (samples x features).
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Add small regularization to avoid numerical issues
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-10
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Select the top n_components eigenvectors if n_components is specified
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
        
        self.components = eigenvectors
        self.explained_variance_ = eigenvalues

    def encode(self, X):
        """
        Reduce the dimensionality of the input data using the learned eigenvectors.

        Parameters:
        X (ndarray): The input data (samples x features).

        Returns:
        ndarray: The reduced-dimensionality data.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def forward(self, X):
        """
        Reconstruct the data from the latent space (reduced data).

        Parameters:
        X (ndarray): The reduced data (latent space representation).

        Returns:
        ndarray: The reconstructed data.
        """
        reconstructed = np.dot(X, self.components.T) + self.mean
        return reconstructed

    def transform(self, X):
        """
        Project data into the lower-dimensional space.

        Parameters:
        X (ndarray): The input data to transform.

        Returns:
        ndarray: The transformed data.
        """
        return self.encode(X)

    def inverse_transform(self, X):
        """
        Reconstruct the data from the lower-dimensional space.

        Parameters:
        X (ndarray): The encoded (reduced-dimensional) data.

        Returns:
        ndarray: The reconstructed data.
        """
        return self.forward(X)

