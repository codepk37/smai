import numpy as np
import matplotlib.pyplot as plt
import os
class KDE:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        """
        Initialize the KDE class.
        Parameters:
        - bandwidth: Smoothing parameter (float)
        - kernel: Kernel type (str), options are 'box', 'gaussian', and 'triangular'
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data = None

    def fit(self, data):
        """
        Fit the model to the input data.
        
        Parameters:
        - data: numpy array of shape (n_samples, n_features)
        """
        self.data = data

    def _kernel(self, distance):
        """
        Kernel function based on the selected kernel type.
        
        Parameters:
        - distance: The distance to apply the kernel on
        
        Returns:
        - Kernel applied on the distance
        """
        if self.kernel == 'box':
            return np.where(np.abs(distance) <= self.bandwidth, 0.5 / self.bandwidth, 0)
        elif self.kernel == 'gaussian':
            return (1 / (self.bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance / self.bandwidth) ** 2)
        elif self.kernel == 'triangular':
            return np.maximum(0, (1 - np.abs(distance / self.bandwidth)) / self.bandwidth)
        else:
            raise ValueError("Unsupported kernel type. Choose 'box', 'gaussian', or 'triangular'.")

    def predict(self, X):
        """
        Predict density estimates at the given points.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        
        Returns:
        - Densities at each input point
        """
        n_samples, n_features = X.shape
        densities = np.zeros(n_samples)
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.data - x, axis=1)
            kernels = self._kernel(distances)
            densities[i] = np.sum(kernels) / (len(self.data) * self.bandwidth ** n_features)
        return densities

    def visualize(self):
        """
        Visualize the density for 2D data.
        """
        if self.data.shape[1] != 2:
            raise ValueError("Visualization only works for 2D data.")

        # Create a grid to evaluate KDE
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]

        # Predict densities at grid points
        densities = self.predict(grid_points).reshape(x_grid.shape)

        # Plot density and data points
        plt.figure(figsize=(8, 6))
        plt.contourf(x_grid, y_grid, densities, cmap='viridis')
        plt.scatter(self.data[:, 0], self.data[:, 1], c='white', s=2, edgecolor='k')
        plt.colorbar(label='Density')
        plt.title(f"KDE Density Estimation (Kernel: {self.kernel})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        plt.savefig("./assignments/5/figures/task2.png")
