import numpy as np

#for encoder
import numpy as np

#her her ehr ehre erereh ehr herer ehr er erhr

class MLPRegressorMultiOutput: #use this in all
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, 
                 activation='relu', optimizer='sgd', batch_size=32, epochs=100):
        """
        Initialize the MLP Regressor with the given hyperparameters.
        
        Parameters:
        - input_size: Number of input features.
        - hidden_layers: List containing the number of neurons in each hidden layer.
        - output_size: Number of output neurons (for multi-output regression).
        - learning_rate: Learning rate for weight updates.
        - activation: Activation function to use.
        - optimizer: Optimization technique to use.
        - batch_size: Number of samples per gradient update.
        - epochs: Number of epochs to train.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function_name = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

        # Initialize weights and biases
        self.weights, self.biases = self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases for the network."""
        layers = [self.input_size] + self.hidden_layers + [self.output_size]  # Full architecture
        print(layers)
        weights = []
        biases = []

        for i in range(len(layers) - 1):
            weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            biases.append(np.random.randn(1, layers[i + 1]))  # Bias should match the next layer's size
        
        return weights, biases

    def _set_activation_function(self, activation):
        """Set the activation function based on the user's choice."""
        if activation == 'sigmoid':
            return self.sigmoid
        elif activation == 'tanh':
            return self.tanh
        elif activation == 'relu':
            return self.relu
        elif activation == 'linear':
            return self.linear
        else:
            raise ValueError("Unsupported activation function.")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def relu(self, z):
        return np.maximum(0, z)

    def linear(self, z):
        """Linear activation function."""
        return z

    def _forward_propagation(self, X):
        """Perform forward propagation through the network."""
        self.activations = [X]  # Store input as the first activation
        self.z_values = []  # To store pre-activation (z) values

        # Forward pass through each layer
        for i in range(len(self.weights)):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            A = self._activate(z) if i < len(self.weights) - 1 else z  # Last layer has no activation
            self.activations.append(A)

        # Output of the network is the final layer activation
        return  self.z_values #self.activations[-1]

    def _activate(self, z):
        """Apply the chosen activation function."""
        if self.activation_function_name == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_function_name == 'tanh':
            return self.tanh(z)
        elif self.activation_function_name == 'relu':
            return self.relu(z)
        elif self.activation_function_name == 'linear':
            return self.linear(z)
        else:
            raise ValueError("Unsupported activation function.")

    def _backward_propagation(self, y):
        """Perform backward propagation to update weights and biases."""
        m = y.shape[0]  # Number of samples
        output_activation = self.activations[-1]
        delta = output_activation - y  # Error term for regression

        # Gradients for the output layer and hidden layers
        gradients = {'weights': [], 'biases': []}
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients['weights'].insert(0, dw)
            gradients['biases'].insert(0, db)
            if i > 0:  # Skip backpropagation for input layer
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i - 1])

        return gradients

    def _activation_derivative(self, z):
        """Calculate the derivative of the activation function."""
        if self.activation_function_name == 'sigmoid':
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.activation_function_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_function_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_function_name == 'linear':
            return np.ones_like(z)
        else:
            raise ValueError("Unsupported activation function.")

    def fit(self, X, y,Val=None):
        """Fit the model (train the regressor) using the input data and labels."""
        best_loss = float('inf')
        patience_counter = 0
        # Store the best weights and biases
        best_weights = None
        best_biases = None    # in case last checkpoint have higher compared error

        for epoch in range(self.epochs):
            if self.optimizer == "batch":
                self.batch_size = X.shape[0]  # Full batch
            elif self.optimizer == "sgd":
                self.batch_size = 1
            indices = np.random.permutation(X.shape[0])  # Shuffle data
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[indices[i:i + self.batch_size]]
                y_batch = y[indices[i:i + self.batch_size]]

                # Forward and backward propagation
                self._forward_propagation(X_batch)
                gradients = self._backward_propagation( y_batch)

                # Update weights and biases
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * gradients['weights'][j]
                    self.biases[j] -= self.learning_rate * gradients['biases'][j]

            # Optionally print loss or other metrics every few epochs
            if epoch % 10 == 0:
                loss = self.compute_loss(y, self.predict(X))
                print(f'Epoch Train {epoch}: Loss = {loss:.4f}')
                if Val is not None:
                    loss = self.compute_loss(Val, self.predict(Val))
                    print(f'Epoch Val {epoch}: Loss = {loss:.4f}')

                #EARLY STOPPING
                if loss < best_loss: 
                    best_loss, patience_counter = loss, 0  # Reset if improvement
                    # Save best weights and biases
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else: patience_counter += 1  # Increment if no improvement

                if patience_counter >= 1000:  # Early stop after 10 consecutive non-improvements
                    print(f"Early stopping at epoch {epoch}")
                    break

        # AT end stopping, restore the best weights and biases/checkpoints
        self.weights = best_weights
        self.biases = best_biases
                    

    def predict(self, X):
        """Make predictions using the trained regressor."""
        output = self._forward_propagation(X)
        return output[-1]  # Return the predicted outputs

    def compute_loss(self, y_true, y_pred):
        
        """Compute Mean Squared Error Loss for regression."""
        return np.mean((y_true - y_pred) ** 2)


    def MSE(self, y, pred):
        """Compute Mean Squared Error for multi-output regression."""
        return np.mean((y - pred)**2, axis=0)  # MSE for each output dimension

    def RMSE(self, y, pred):
        """Compute Root Mean Squared Error for multi-output regression."""
        return np.sqrt(np.mean((y - pred)**2, axis=0))  # RMSE for each output dimension

    def Rsquared(self, y, pred):
        """Compute R-squared for multi-output regression."""
        ss_res = np.sum((y - pred)**2, axis=0)  # Residual sum of squares for each output
        ss_tot = np.sum((y - np.mean(y, axis=0))**2, axis=0)  # Total sum of squares for each output
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared  # Returns R^2 for each output dimension

    #CHATGPT PRODUCED THIS
    def gradient_checking(self, X, y, epsilon=1e-7):
        """
        Perform gradient checking to verify the accuracy of backpropagation.
        
        Args:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): True labels
        epsilon (float): Small value for computing numerical gradient
        
        Returns:
        float: Relative error between numerical and analytical gradients
        """
        # Compute analytical gradients
        self._forward_propagation(X)
        analytical_gradients = self._backward_propagation(y)
        
        # Compute numerical gradients
        numerical_gradients = {'weights': [], 'biases': []}
        
        for l in range(len(self.weights)):
            numerical_gradients['weights'].append(np.zeros_like(self.weights[l]))
            numerical_gradients['biases'].append(np.zeros_like(self.biases[l]))
            
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    # Compute J(θ + ε)
                    self.weights[l][i, j] += epsilon
                    self._forward_propagation(X)
                    cost_plus = self.compute_loss(y, self.activations[-1])
                    
                    # Compute J(θ - ε)
                    self.weights[l][i, j] -= 2 * epsilon
                    self._forward_propagation(X)
                    cost_minus = self.compute_loss(y, self.activations[-1])
                    
                    # Restore original weight
                    self.weights[l][i, j] += epsilon
                    
                    # Compute numerical gradient
                    numerical_gradients['weights'][l][i, j] = (cost_plus - cost_minus) / (2 * epsilon)
            
            for i in range(self.biases[l].shape[1]):
                # Compute J(θ + ε)
                self.biases[l][0, i] += epsilon
                self._forward_propagation(X)
                cost_plus = self.compute_loss(y, self.activations[-1])
                
                # Compute J(θ - ε)
                self.biases[l][0, i] -= 2 * epsilon
                self._forward_propagation(X)
                cost_minus = self.compute_loss(y, self.activations[-1])
                
                # Restore original bias
                self.biases[l][0, i] += epsilon
                
                # Compute numerical gradient
                numerical_gradients['biases'][l][0, i] = (cost_plus - cost_minus) / (2 * epsilon)
        
        # Compute relative error
        total_error = 0
        total_norm = 0
        for l in range(len(self.weights)):
            weight_error = np.mean(numerical_gradients['weights'][l] - analytical_gradients['weights'][l])
            weight_norm = np.mean(numerical_gradients['weights'][l]) + np.linalg.norm(analytical_gradients['weights'][l])
            bias_error = np.mean(numerical_gradients['biases'][l] - analytical_gradients['biases'][l])
            bias_norm = np.mean(numerical_gradients['biases'][l]) + np.linalg.norm(analytical_gradients['biases'][l])
            
            total_error += weight_error + bias_error
            total_norm += weight_norm + bias_norm
        
        relative_error = total_error / total_norm
        return relative_error



class MLPRegressorMultiOutputWithLoss(MLPRegressorMultiOutput):
    def __init__(self, input_size, hidden_layers, output_size=1, learning_rate=0.01,
                 activation='relu', optimizer='sgd', batch_size=32, epochs=100, loss_function='bce'):
        """
        Initialize the MLP Regressor with the given hyperparameters, extending it to use different loss functions.

        Parameters:
        - input_size: Number of input features.
        - hidden_layers: List containing the number of neurons in each hidden layer.
        - output_size: Set to 1 for binary classification (logistic regression).
        - learning_rate: Learning rate for weight updates.
        - activation: Activation function to use for hidden layers.
        - optimizer: Optimization technique to use.
        - batch_size: Number of samples per gradient update.
        - epochs: Number of epochs to train.
        - loss_function: Loss function to use ('bce' for Binary Cross-Entropy, 'mse' for Mean Squared Error).
        """
        super().__init__(input_size, hidden_layers, output_size, learning_rate, activation, optimizer, batch_size, epochs)
        self.loss_function = loss_function  # New parameter to select the loss function

    def compute_loss(self, y_true, y_pred):
        """Compute the chosen loss function: Binary Cross-Entropy (BCE) or Mean Squared Error (MSE)."""
        if self.loss_function == 'bce':
            # Binary Cross-Entropy Loss
            epsilon = 1e-15  # To prevent log(0) issues
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping the predictions
            bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return bce_loss
        elif self.loss_function == 'mse':
            # Mean Squared Error Loss
            return np.mean((y_true - y_pred) ** 2)
        else:
            raise ValueError("Unsupported loss function. Choose either 'bce' or 'mse'.")

    def _forward_propagation(self, X):
        """
        Perform forward propagation with the output layer using sigmoid activation for logistic regression.
        """
        self.activations = [X]  # Store input as the first activation
        self.z_values = []  # To store pre-activation (z) values

        # Forward pass through each layer
        for i in range(len(self.weights)):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            # Apply activation to all hidden layers, sigmoid to the output layer
            if i == len(self.weights) - 1:
                A = self.sigmoid(z)  # Sigmoid in the output layer for logistic regression
            else:
                A = self._activate(z)  # Activation for hidden layers
            self.activations.append(A)

        # Output of the network is the final layer activation
        return self.activations[-1]

    def predict(self, X):
        """
        Predict the output using the forward propagation.
        For logistic regression, output will be the sigmoid of the final layer (between 0 and 1).
        """
        output = self._forward_propagation(X)
        return output

    def evaluate(self, X, y_true):
        """
        Evaluate the model by computing predictions and loss on given data.
        """
        y_pred = self.predict(X)
        loss = self.compute_loss(y_true, y_pred)
        return loss








# class MLPRegressorold:
#     def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01,
#                  activation='relu', optimizer='sgd', batch_size=32, epochs=100):
#         """
#         Initialize the MLP model with the given hyperparameters.
        
#         Parameters:
#         - input_size: Number of input features.
#         - hidden_layers: List containing the number of neurons in each hidden layer.
#         - output_size: Number of output neurons (1 for regression).
#         - learning_rate: Learning rate for weight updates.
#         - activation: Activation function to use.
#         - optimizer: Optimization technique to use.
#         - batch_size: Number of samples per gradient update.
#         - epochs: Number of epochs to train.
#         """
#         self.input_size = input_size
#         self.hidden_layers = hidden_layers
#         self.output_size = output_size
#         self.learning_rate = learning_rate
#         self.activation_function_name = activation
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.epochs = epochs

#         # Initialize weights and biases
#         self.weights, self.biases = self._initialize_parameters()

#     def _initialize_parameters(self):
#         layers = [self.input_size] + self.hidden_layers + [self.output_size]
#         weights = []
#         biases = []
#         print(layers)
        
#         for i in range(len(layers) - 1):
#             # Ensure weights have the correct shape
#             weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
#             biases.append(np.random.randn(1, layers[i + 1]))  # Bias should match the next layer's size
        
#         return weights, biases


#     def _set_activation_function(self, activation):
#         """Set the activation function based on the user's choice."""
#         if activation == 'sigmoid':
#             return self.sigmoid
#         elif activation == 'tanh':
#             return self.tanh
#         elif activation == 'relu':
#             return self.relu
#         elif activation== 'linear':
#             return self.linear
#         else:
#             raise ValueError("Unsupported activation function.")

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def tanh(self, z):
#         return np.tanh(z)

#     def relu(self, z):
#         return np.maximum(0, z)
    
#     def linear(self, z):
#         """Linear activation function."""
#         return z

#     def _forward_propagation(self, X):
#         """Perform forward propagation through the network."""
#         self.activations = [X]  # Store input as the first activation
#         self.z_values = []  # To store pre-activation (z) values
        
#         # Forward pass through hidden layers
#         for i in range(len(self.weights) - 1):
#             z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
#             self.z_values.append(z)
#             A = self._activate(z)  # Apply activation function
#             self.activations.append(A)

#         # Output layer , no activations used in last layer, maintain linarity
#         z_output = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
#         self.z_values.append(z_output)
#         A_output = z_output  # No activation for regression output
#         self.activations.append(A_output)

#         return A_output

#     def _activate(self, z):
#         """Apply the chosen activation function."""
#         if self.activation_function_name == 'sigmoid':
#             return self.sigmoid(z)
#         elif self.activation_function_name == 'tanh':
#             return self.tanh(z)
#         elif self.activation_function_name == 'relu':
#             return self.relu(z)
#         elif self.activation_function_name == 'linear':
#             return self.linear(z)
        
#         else:
#             raise ValueError("Unsupported activation function.")

#     def _backward_propagation(self, y):
#         y = y.reshape(-1, 1) #(50,) -> (50,1)
#         """Perform backward propagation to update weights and biases."""
#         m = y.shape[0]  # Number of samples
#         output_activation = self.activations[-1]
#         delta = output_activation - y  # For regression loss
#         # (50,1)

#         gradients = {'weights': [], 'biases': []}
#         for i in reversed(range(len(self.weights))):
#             dw = np.dot(self.activations[i].T, delta) / m
#             db = np.sum(delta, axis=0, keepdims=True) / m
            
#             gradients['weights'].insert(0, dw)
#             gradients['biases'].insert(0, db)
            
#             if i > 0:
#                 delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i - 1])

#         return gradients

#     def _activation_derivative(self, z):
#         """Calculate the derivative of the activation function."""
#         if self.activation_function_name == 'sigmoid':
#             return self.sigmoid(z) * (1 - self.sigmoid(z))
#         elif self.activation_function_name == 'tanh':
#             return 1 - np.tanh(z) ** 2
#         elif self.activation_function_name == 'relu':
#             return (z > 0).astype(float)
#         elif self.activation_function_name == 'linear':
#             return np.ones_like(z)
#         else:
#             raise ValueError("Unsupported activation function.")

#     def fit(self, X, y):
#         best_loss = float('inf') 
#         patience_counter = 0
#         # Store the best weights and biases
#         best_weights = None
#         best_biases = None          # in case last checkpoint have higher compared error
        
#         """Fit the model to the training data."""
#         for epoch in range(self.epochs):
#             if self.optimizer == "batch":
#                 self.batch_size = X.shape[0]  # Full batch
#             elif self.optimizer == "sgd":
#                 self.batch_size = 1 
#             indices = np.random.permutation(X.shape[0])  # Shuffle data
#             for i in range(0, X.shape[0], self.batch_size):
#                 X_batch = X[indices[i:i + self.batch_size]]
#                 y_batch = y[indices[i:i + self.batch_size]]

#                 # Forward and backward propagation
#                 self._forward_propagation(X_batch)
#                 gradients = self._backward_propagation(y_batch)

#                 # Update weights and biases
#                 for j in range(len(self.weights)):
#                     self.weights[j] -= self.learning_rate * gradients['weights'][j]
#                     self.biases[j] -= self.learning_rate * gradients['biases'][j]

#             # Optionally print loss or other metrics every few epochs
#             if epoch % 10 == 0:
#                 loss = self.compute_loss(y, self.predict(X))
#                 print(f'Epoch {epoch}: Loss = {loss:.4f}')

#                 #EARLY STOPPING
#                 if loss < best_loss: 
#                     best_loss, patience_counter = loss, 0  # Reset if improvement
#                     # Save best weights and biases
#                     best_weights = [w.copy() for w in self.weights]
#                     best_biases = [b.copy() for b in self.biases]
#                 else: patience_counter += 1  # Increment if no improvement

#                 if patience_counter >= 1000:  # Early stop after 10 consecutive non-improvements
#                     print(f"Early stopping at epoch {epoch}")
#                     break

#         # AT end stopping, restore the best weights and biases/checkpoints
#         self.weights = best_weights
#         self.biases = best_biases

#     def predict(self, X):
#         """Make predictions using the trained model."""
#         output = self._forward_propagation(X)
#         return output  # For regression, return raw output values

#     def compute_loss(self, y_true, y_pred):
#         """Compute Mean Squared Error Loss."""
#         return np.mean((y_true - y_pred) ** 2)
    
#     def MSE(self,y, pred):
#         return np.mean((y - pred)**2)
#     def RMSE(self,y, pred):
#         return np.sqrt(np.mean((y - pred)**2))
#     def Rsquared(self,y, pred):
#         return 1 - (np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2))

#     def gradient_check(self, X, y, epsilon=1e-4):
#         """Perform gradient checking."""
#         num_gradients = []
#         for i in range(len(self.weights)):
#             original_weights = self.weights[i].copy()
#             gradient = np.zeros_like(original_weights)

#             for j in range(original_weights.size):
#                 original_weights[j] += epsilon
#                 loss_plus = self.compute_loss(y, self.predict(X))
                
#                 original_weights[j] -= 2 * epsilon
#                 loss_minus = self.compute_loss(y, self.predict(X))
                
#                 gradient[j] = (loss_plus - loss_minus) / (2 * epsilon)
#                 original_weights[j] += epsilon  # Restore original weight
            
#             num_gradients.append(gradient.reshape(original_weights.shape))
        
#         analytical_gradients = self._backward_propagation(y)
#         for i in range(len(num_gradients)):
#             if not np.allclose(num_gradients[i], analytical_gradients[i], rtol=1e-4, atol=1e-4):
#                 print(f"Gradient check failed for layer {i}")
#                 return False
        
#         print("Gradient check passed.")
#         return True











class MultiLabelMLP:
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.01, activation='relu', 
                 output_activation='sigmoid', loss='binary_cross_entropy',
                 batch_size=32, epochs=100):
        """
        Initialize the MLP model for multi-label classification.

        Parameters:
        - input_size: Number of input features.
        - hidden_layers: List containing the number of neurons in each hidden layer.
        - output_size: Number of output neurons (equal to the number of classes).
        - learning_rate: Learning rate for weight updates.
        - activation: Activation function to use for hidden layers.
        - output_activation: Activation function to use for output layer (sigmoid).
        - loss: Loss function (binary cross-entropy for multi-label classification).
        - batch_size: Number of samples per gradient update.
        - epochs: Number of epochs to train the model.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function_name = activation
        self.activation_function = self._set_activation_function(activation)
        self.output_activation_function_name = output_activation
        self.output_activation_function = self._set_activation_function(output_activation)
        self.loss_function_name = loss
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize weights and biases
        self.weights, self.biases = self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases for each layer."""
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        weights = []
        biases = []
        
        for i in range(len(layers) - 1):
            weights.append(np.random.randn(layers[i], layers[i+1]) * 0.2)
            biases.append(np.zeros((1, layers[i+1])))
        
        return weights, biases
    
    def _set_activation_function(self, activation):
        """Set the activation function based on the user's choice."""
        if activation == 'sigmoid':
            return self.sigmoid
        elif activation == 'tanh':
            return self.tanh
        elif activation == 'relu':
            return self.relu
        elif activation == 'linear':
            return self.linear
        else:
            raise ValueError("Unsupported activation function.")
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))
    
    def tanh(self, z):
        """Tanh activation function."""
        return np.tanh(z)
    
    def relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def linear(self, z):
        """Linear activation function."""
        return z
    
    def binary_cross_entropy_loss(self, y_true, y_pred):
        """Compute Binary Cross-Entropy Loss."""
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    
    def _forward_propagation(self, X):
        """Perform forward propagation through the network."""
        self.activations = [X]  # Store input as the first activation
        self.z_values = []  # To store pre-activation (z) values
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):  # Skip the last layer (output layer)
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            A = self.activation_function(z)  # Apply activation function to hidden layers
            self.activations.append(A)
        
        # Output layer
        z_output = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        
        # Apply sigmoid for multi-label classification
        A_output = self.sigmoid(z_output)
        self.activations.append(A_output)
        return A_output

    
    def _backward_propagation(self, y):
        """Perform backward propagation to update weights and biases."""
        m = y.shape[0]
        output_activation = self.activations[-1]
        delta = output_activation - y  # Error at the output layer for binary classification
        
        gradients = {'weights': [], 'biases': []}
        
        # Backpropagation through the layers
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients['weights'].insert(0, dw)  # Insert at the beginning to reverse order
            gradients['biases'].insert(0, db)
            
            if i > 0:  # Update delta for hidden layers
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i-1])
        
        return gradients
    
    def _activation_derivative(self, z):
        """Calculate the derivative of the activation function."""
        if self.activation_function_name == 'sigmoid':
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.activation_function_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_function_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_function_name == 'linear':
            return np.ones_like(z)
    
    def compute_loss(self, y_true, y_pred):
        """Compute the loss based on binary cross-entropy."""
        return self.binary_cross_entropy_loss(y_true, y_pred)

    def fit(self, X, y):
        """Fit the model to the training data."""
        for epoch in range(self.epochs):
            # Shuffle the data for mini-batch gradient descent
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Forward and backward propagation
                self._forward_propagation(X_batch)
                gradients = self._backward_propagation(y_batch)
                
                # Update weights and biases
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * gradients['weights'][j]
                    self.biases[j] -= self.learning_rate * gradients['biases'][j]

            if epoch % 10 == 0:
                loss = self.compute_loss(y, self._forward_propagation(X))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        """Make predictions using the trained model."""
        output = self._forward_propagation(X)
        return (output > 0.5).astype(int)  # Convert probabilities to binary 0 or 1


    def gradient_checking(self, X, y, epsilon=1e-5):
        """
        Perform gradient checking to verify the accuracy of backpropagation.
        
        Args:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): True labels
        epsilon (float): Small value for computing numerical gradient
        
        Returns:
        float: Relative error between numerical and analytical gradients
        """
        # Compute analytical gradients
        self._forward_propagation(X)
        analytical_gradients = self._backward_propagation(y)
        
        # Compute numerical gradients
        numerical_gradients = {'weights': [], 'biases': []}
        
        for l in range(len(self.weights)):
            numerical_gradients['weights'].append(np.zeros_like(self.weights[l]))
            numerical_gradients['biases'].append(np.zeros_like(self.biases[l]))
            
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    # Compute J(θ + ε)
                    self.weights[l][i, j] += epsilon
                    self._forward_propagation(X)
                    cost_plus = self.compute_loss(y, self.activations[-1])
                    
                    # Compute J(θ - ε)
                    self.weights[l][i, j] -= 2 * epsilon
                    self._forward_propagation(X)
                    cost_minus = self.compute_loss(y, self.activations[-1])
                    
                    # Restore original weight
                    self.weights[l][i, j] += epsilon
                    
                    # Compute numerical gradient
                    numerical_gradients['weights'][l][i, j] = (cost_plus - cost_minus) / (2 * epsilon)
            
            for i in range(self.biases[l].shape[1]):
                # Compute J(θ + ε)
                self.biases[l][0, i] += epsilon
                self._forward_propagation(X)
                cost_plus = self.compute_loss(y, self.activations[-1])
                
                # Compute J(θ - ε)
                self.biases[l][0, i] -= 2 * epsilon
                self._forward_propagation(X)
                cost_minus = self.compute_loss(y, self.activations[-1])
                
                # Restore original bias
                self.biases[l][0, i] += epsilon
                
                # Compute numerical gradient
                numerical_gradients['biases'][l][0, i] = (cost_plus - cost_minus) / (2 * epsilon)
        
        # Compute relative error
        error = 0
        for l in range(len(self.weights)):
            error += np.linalg.norm(numerical_gradients['weights'][l] - analytical_gradients['weights'][l]) / \
                     (np.linalg.norm(numerical_gradients['weights'][l]) + np.linalg.norm(analytical_gradients['weights'][l]))
            error += np.linalg.norm(numerical_gradients['biases'][l] - analytical_gradients['biases'][l]) / \
                     (np.linalg.norm(numerical_gradients['biases'][l]) + np.linalg.norm(analytical_gradients['biases'][l]))
        
        return error / (2 * len(self.weights))    






import numpy as np

class MLP: #single lable classification  
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.01, activation='relu', 
                 output_activation='softmax', loss='cross_entropy',
                 batch_size=50, epochs=100, optimizer="mini-batch"):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function_name = activation
        self.activation_function = self._set_activation_function(activation)
        self.output_activation_function_name = output_activation
        self.loss_function_name = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer

        self.weights, self.biases = self._initialize_parameters()
    
    def _initialize_parameters(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        print(layers)
        weights = []
        biases = []
        
        for i in range(len(layers) - 1):
            weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2. / layers[i]))  # He initialization
            biases.append(np.zeros((1, layers[i+1])))
    
        return weights, biases
    
    def _set_activation_function(self, activation):
        if activation == 'sigmoid':
            return self.sigmoid
        elif activation == 'tanh':
            return self.tanh
        elif activation == 'relu':
            return self.relu
        elif activation == 'linear':
            return self.linear
        else:
            raise ValueError("Unsupported activation function.")
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -709, 709)))  # Clip to avoid overflow
    
    def tanh(self, z):
        return np.tanh(z)
    
    def relu(self, z):
        return np.maximum(0, z)

    def linear(self, z):
        return z
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    
    def _forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            A = self.activation_function(z)
            self.activations.append(A)
        
        z_output = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        
        if self.output_activation_function_name == 'softmax':
            A_output = self.softmax(z_output)
        else:
            raise ValueError("Unsupported output activation function.")
        
        self.activations.append(A_output)
        return A_output

    def _backward_propagation(self, y):
        m = y.shape[0]
        gradients = {'weights': [], 'biases': []}
        
        # Output layer
        delta = self.activations[-1] - y
        
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients['weights'].insert(0, dw)
            gradients['biases'].insert(0, db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i-1])

        return gradients
    
    def _activation_derivative(self, z):
        if self.activation_function_name == 'sigmoid':
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.activation_function_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_function_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_function_name == 'linear':
            return np.ones_like(z)

    def fit(self, X, y, X_val=None, y_val=None, max_patience=10):
        best_loss = float('inf')
        patience_counter = 0
        best_weights, best_biases = None, None

        for epoch in range(self.epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            if self.optimizer == "batch":
                self.batch_size = X.shape[0]
            elif self.optimizer == "sgd":
                self.batch_size = 1
            
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                self._forward_propagation(X_batch)
                gradients = self._backward_propagation(y_batch)

                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * gradients['weights'][j]
                    self.biases[j] -= self.learning_rate * gradients['biases'][j]

            if epoch % 10 == 0:
                predictions = self.predict(X)
                loss = self.compute_loss(y, predictions)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

                if X_val is not None and y_val is not None:
                    val_predictions = self.predict(X_val)
                    val_loss = self.compute_loss(y_val, val_predictions)
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                    else:
                        patience_counter += 1

                    if patience_counter >= max_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        if best_weights is not None and best_biases is not None:
            self.weights = best_weights
            self.biases = best_biases

    def predict(self, X):
        return self._forward_propagation(X)

    def compute_loss(self, y_true, y_pred):
        if self.loss_function_name == 'cross_entropy':
            return self.cross_entropy_loss(y_true, y_pred)

    
    def gradient_checking(self, X, y, epsilon=1e-5):
        """
        Perform gradient checking to verify the accuracy of backpropagation.
        
        Args:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): True labels
        epsilon (float): Small value for computing numerical gradient
        
        Returns:
        float: Relative error between numerical and analytical gradients
        """
        # Compute analytical gradients
        self._forward_propagation(X)
        analytical_gradients = self._backward_propagation(y)
        
        # Compute numerical gradients
        numerical_gradients = {'weights': [], 'biases': []}
        
        for l in range(len(self.weights)):
            numerical_gradients['weights'].append(np.zeros_like(self.weights[l]))
            numerical_gradients['biases'].append(np.zeros_like(self.biases[l]))
            
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    # Compute J(θ + ε)
                    self.weights[l][i, j] += epsilon
                    self._forward_propagation(X)
                    cost_plus = self.compute_loss(y, self.activations[-1])
                    
                    # Compute J(θ - ε)
                    self.weights[l][i, j] -= 2 * epsilon
                    self._forward_propagation(X)
                    cost_minus = self.compute_loss(y, self.activations[-1])
                    
                    # Restore original weight
                    self.weights[l][i, j] += epsilon
                    
                    # Compute numerical gradient
                    numerical_gradients['weights'][l][i, j] = (cost_plus - cost_minus) / (2 * epsilon)
            
            for i in range(self.biases[l].shape[1]):
                # Compute J(θ + ε)
                self.biases[l][0, i] += epsilon
                self._forward_propagation(X)
                cost_plus = self.compute_loss(y, self.activations[-1])
                
                # Compute J(θ - ε)
                self.biases[l][0, i] -= 2 * epsilon
                self._forward_propagation(X)
                cost_minus = self.compute_loss(y, self.activations[-1])
                
                # Restore original bias
                self.biases[l][0, i] += epsilon
                
                # Compute numerical gradient
                numerical_gradients['biases'][l][0, i] = (cost_plus - cost_minus) / (2 * epsilon)
        
        # Compute relative error
        error = 0
        for l in range(len(self.weights)):
            error += np.linalg.norm(numerical_gradients['weights'][l] - analytical_gradients['weights'][l]) / \
                     (np.linalg.norm(numerical_gradients['weights'][l]) + np.linalg.norm(analytical_gradients['weights'][l]))
            error += np.linalg.norm(numerical_gradients['biases'][l] - analytical_gradients['biases'][l]) / \
                     (np.linalg.norm(numerical_gradients['biases'][l]) + np.linalg.norm(analytical_gradients['biases'][l]))
        
        return error / (2 * len(self.weights))    






import numpy as np

class CommonMLP_bonus:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01,
                 activation='relu', output_activation='linear', loss='mse', 
                 batch_size=50, epochs=100, optimizer='mini-batch', task_type='regression'):
        """
        Generalized MLP for both classification and regression tasks.
        
        Parameters:
        - input_size: Number of input features.
        - hidden_layers: List containing the number of neurons in each hidden layer.
        - output_size: Number of output neurons (1 for regression, number of classes for classification).
        - learning_rate: Learning rate for weight updates.
        - activation: Activation function to use for hidden layers.
        - output_activation: Activation function for output layer ('softmax' for classification, 'linear' for regression).
        - loss: Loss function ('cross_entropy', 'mse', etc.).
        - batch_size: Number of samples per gradient update.
        - epochs: Number of epochs to train.
        - optimizer: Optimization method ('sgd', 'batch', 'mini-batch').
        - task_type: 'regression' or 'classification'.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function_name = activation
        self.activation_function = self._set_activation_function(activation)
        self.output_activation_function_name = output_activation
        self.loss_function_name = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.task_type = task_type
        
        # Initialize weights and biases
        self.weights, self.biases = self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initialize weights and biases for each layer.
        """
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        print("layers:", layers)
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)  # He initialization
            biases.append(np.zeros((1, layers[i+1])))
        return weights, biases

    def _set_activation_function(self, activation):
        """
        Set the activation function for hidden layers.
        """
        if activation == 'sigmoid':
            return self.sigmoid
        elif activation == 'tanh':
            return self.tanh
        elif activation == 'relu':
            return self.relu
        elif activation == 'linear':
            return self.linear
        else:
            raise ValueError("Unsupported activation function.")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def relu(self, z):
        return np.maximum(0, z)

    def linear(self, z):
        return z

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        """
        self.activations = [X]  # Store input as the first activation
        self.z_values = []  # To store pre-activation (z) values
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            A = self.activation_function(z)  # Apply hidden layer activation
            self.activations.append(A)
        
        # Output layer
        z_output = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        
        # Apply the correct output activation function
        if self.output_activation_function_name == 'softmax':
            A_output = self.softmax(z_output)  # Classification with softmax
        elif self.output_activation_function_name == 'linear':
            A_output = self.linear(z_output)  # Regression with linear
        else:
            raise ValueError("Unsupported output activation function.")
        
        self.activations.append(A_output)
        return A_output

    def _backward_propagation(self, y):
        """
        Perform backward propagation to update weights and biases.
        """
        m = y.shape[0]
        output_activation = self.activations[-1]
        delta = output_activation - y  # Generalized delta (for both regression and classification)

        gradients = {'weights': [], 'biases': []}
        
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients['weights'].insert(0, dw)
            gradients['biases'].insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i - 1])

        return gradients

    def _activation_derivative(self, z):
        """
        Calculate the derivative of the activation function for hidden layers.
        """
        if self.activation_function_name == 'sigmoid':
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.activation_function_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_function_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_function_name == 'linear':
            return np.ones_like(z)

    def compute_loss(self, y_true, y_pred):
        """
        Compute loss based on the task type and the selected loss function.
        """
        if self.loss_function_name == 'mse' and self.task_type == 'regression':
            return np.mean((y_true - y_pred) ** 2)  # Mean Squared Error for regression
        elif self.loss_function_name == 'cross_entropy' and self.task_type == 'classification':
            return -np.mean(y_true * np.log(y_pred + 1e-8))  # Cross-Entropy for classification
        else:
            raise ValueError("Unsupported loss function.")

    def fit(self, X, y, max_patience=10):
        """
        Train the model using mini-batch gradient descent with early stopping.
        """
        best_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(self.epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            if self.optimizer == 'batch':
                self.batch_size = X.shape[0]  # Full batch
            elif self.optimizer == 'sgd':
                self.batch_size = 1  # Stochastic

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Forward and backward propagation
                self._forward_propagation(X_batch)
                gradients = self._backward_propagation(y_batch)

                # Update weights and biases
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * gradients['weights'][j]
                    self.biases[j] -= self.learning_rate * gradients['biases'][j]

            if epoch % 10 == 0:
                y_pred = self.predict(X)
                loss = self.compute_loss(y, y_pred)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

                if loss < best_loss:
                    best_loss, patience_counter = loss, 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.weights = best_weights
        self.biases = best_biases

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        output = self._forward_propagation(X)
        if self.output_activation_function_name == 'softmax':
            return output  #return np.argmax(output, axis=1)  # Classification: Return class indices
        elif self.output_activation_function_name == 'linear':
            return output  # Regression: Return raw output values

