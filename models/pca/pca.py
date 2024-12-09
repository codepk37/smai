import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.X_reduced = None  # To store the reduced data for validation
        self.explained_variance_ = None  # To store eigenvalues (variance of each component)

    def fit(self, X):
        # Ensure n_components is less than or equal to the number of features
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot be greater than the number of features")

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
        
        # Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # Store eigenvalues
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Apply PCA transformation and store reduced data
        self.X_reduced = self.transform(X)  

    def transform(self, X):
        # Check if fit() has been called
        if self.components is None:
            raise RuntimeError("PCA has not been fitted yet. Call fit() before transform().")
        
        # Center the data
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        return np.dot(X_centered, self.components)


    def explained_variance_ratio(self):
        # Calculate explained variance ratio
        total_variance = np.sum(self.explained_variance_)
        return self.explained_variance_ / total_variance
    
    def checkPCA(self):
        # Check if fit() has been called
        if self.components is None or self.X_reduced is None:
            raise RuntimeError("PCA has not been fitted yet. Call fit() before checkPCA().")

        # Step 1: Check if the number of dimensions is reduced to n_components
        is_reduced_dimension_correct = self.X_reduced.shape[1] == self.n_components

        # Step 2: Reconstruct the data from the reduced dimensions
        X_reconstructed = np.dot(self.X_reduced, self.components.T) + self.mean

        # Step 3: Check reconstruction accuracy using Mean Squared Error (MSE)
        reconstruction_error = np.mean((X_reconstructed - (self.mean + (self.X_reduced @ self.components.T)))**2)

        threshold =0.1
        # Return the check results
        return reconstruction_error<threshold
