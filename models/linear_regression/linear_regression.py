import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

   
class PolynomialLinearRegression:
    def __init__(self, degree, learning_rate=0.01, max_epochs=1000, tolerance=1e-6,regularization_type=None, lambda_=0.01):
        self.degree = degree
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.coefficients = None
        self.convergence_epoch = None  # Add attribute to store convergence epoch
        self.history = []  # To store the coefficients at each epoch .gif curve of data doing gradiend descent i.e. training data
        self.regularization_type = regularization_type #Type of regularization ('L1', 'L2', or None)
        self.lambda_ = lambda_ # Regularization strength

    def fit(self, X_train, y_train):
        """
        Fit the polynomial linear regression model to the training data.
        X_train: feature vector
        y_train: target vector
        """
        # Prepare polynomial features
        X_poly = np.ones((len(X_train), self.degree + 1))
        for i in range(1, self.degree + 1):
            X_poly[:, i] = X_train ** i #single x colum x1
        #wont be updated again

        # Initialize coefficients
        self.coefficients = np.random.rand(X_poly.shape[1]) #degree+1,  1 b0 b1 b2 .. (range 0.0,1.0)
        self.convergence_epoch=self.max_epochs
        
        N = len(y_train)
        
        for epoch in range(self.max_epochs):
            # Compute predictions
            y_pred = np.dot(X_poly, self.coefficients) #cor
            
            # Compute the error
            error = y_train - y_pred #cor (320,)
        
            
            # Compute gradients
            gradients = -2 * np.dot(X_poly.T, error) / N #cor

            # Apply regularization ,only this is added
            if self.regularization_type == 'L1':
                gradients += self.lambda_ * np.sign(self.coefficients)
            elif self.regularization_type == 'L2':
                gradients += 2 * self.lambda_ * self.coefficients

            # Update coefficients
            new_coefficients = self.coefficients - self.learning_rate * gradients #cor
            
            # Store the coefficients for animation
            self.history.append((epoch, self.mse(y_pred,y_train), np.var(y_pred), np.std(y_pred), np.copy(self.coefficients)))

            # Check for convergence
            if np.all(np.abs(new_coefficients - self.coefficients) < self.tolerance):
                # print(f"Converged at epoch {epoch}")
                self.convergence_epoch = epoch  # Store the epoch where convergence happened
                break
                
            self.coefficients = new_coefficients

    def predict(self, X):
        """
        Predict the target values for the input data X.
        """
        # Prepare polynomial features for prediction
        X_poly = np.ones((len(X), self.degree + 1))
        for i in range(1, self.degree + 1):
            X_poly[:, i] = X ** i
        
        return np.dot(X_poly, self.coefficients)
    
    def get_coefficients(self):
        return self.coefficients
    
    def epoch_converged(self):
        return self.convergence_epoch
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def variance(self, y_true):
        return np.var(y_true)

    def std_dev(self, y_true):
        return np.std(y_true)
    
    def plot(self, X, y):
        plt.figure(figsize=(10, 6))
        
        # Plotting the test data
        plt.scatter(X, y, color='blue', label='Data Points')
        
        # Generating prediction curve
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred = self.predict(x_range) #uses predict method to send x_range as input, have already access to self.coefficients
        
        # Plotting the regression curve
        plt.plot(x_range, y_pred, color='red', label='Fitted Curve')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression Fit')
        plt.legend()
        # plt.savefig(f'./assignments/1/figures/lin_degree{self.degree}.png')
        plt.show()





    def create_animation(self, X_train, y_train, save_path='assignments/1/figures/convergence.gif'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        max_frames=150 # that only 150 frames are used in the GIF,
        total_frames = len(self.history) 
        if total_frames > max_frames:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int) #to select 150 equally spaced indices from the self.history
            # print(indices)
        else:
            indices = np.arange(total_frames)
        
        def update(i):
            index = indices[i]
            epoch, mse, variance, std_dev, coefficients = self.history[index]
            y_pred = self.predict(X_train)

            # Clear previous plots
            for ax in axes.flatten():
                ax.clear()

            # Plotting the regression curve
            axes[0, 0].scatter(X_train, y_train, color='blue', label='Data Points')
            x_range = np.linspace(X_train.min(), X_train.max(), 100)
            y_range_pred = sum(coefficients[j] * (x_range ** j) for j in range(self.degree + 1)) #1-1000 selected indices
            axes[0, 0].plot(x_range, y_range_pred, color='red', label='Fitted Curve')
            axes[0, 0].set_title(f'Epoch: {epoch}, MSE: {mse:.4f}')
            axes[0, 0].legend()

            # Plotting MSE
            axes[0, 1].plot(range(i+1), [h[1] for h in self.history[:i+1]], color='orange') #1-150 only, of start
            axes[0, 1].set_title(f'MSE over Epochs')
            axes[0, 1].set_ylabel('MSE')

            # Plotting Variance
            axes[1, 0].plot(range(i+1), [h[2] for h in self.history[:i+1]], color='green')
            axes[1, 0].set_title(f'Variance over Epochs')
            axes[1, 0].set_ylabel('Variance')

            # Plotting Standard Deviation
            axes[1, 1].plot(range(i+1), [h[3] for h in self.history[:i+1]], color='blue')
            axes[1, 1].set_title(f'Standard Deviation over Epochs')
            axes[1, 1].set_ylabel('Standard Deviation')
        
        anim = animation.FuncAnimation(fig, update, frames=len(indices), repeat=False)
        anim.save(save_path, writer='pillow', fps=60)
        plt.close(fig)
        print("Saved .gif")
