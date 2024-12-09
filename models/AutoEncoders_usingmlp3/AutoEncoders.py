import numpy as np
from models.MLP.MLP import *

class AutoEncoder:
    def __init__(self, input_size, before,latent_size,after,output_size, learning_rate=0.01, 
                 activation='relu', optimizer='sgd', batch_size=50, epochs=100):
        """
        Initialize the AutoEncoder model.
        
        Parameters:
        - input_size: Number of input features (same as original data).
        - latent_size: Number of dimensions to reduce the data to.
        - learning_rate: Learning rate for weight updates.
        - activation: Activation function to use (e.g., relu, sigmoid).
        - optimizer: Optimization technique to use (e.g., sgd).
        - batch_size: Number of samples per gradient update.
        - epochs: Number of epochs to train.
        """
        self.input_size = input_size
        self.before      =before 
        self.latent_size = latent_size
        self.after       = after
        self.output_size = output_size  
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.mlp=  MLPRegressorMultiOutput(input_size=self.input_size,
                    hidden_layers=self.before+[self.latent_size]+self.after,
                    output_size=self.output_size,
                    learning_rate=self.learning_rate,
                    activation=self.activation,
                    optimizer=self.optimizer,
                    batch_size=self.batch_size,
                    epochs=self.epochs
                    )
   

    def fit(self,X,y,Val=None):
        """
        Train the AutoEncoder model using forward and backward passes.
        
        Parameters:
        - X: Input data to train the autoencoder.
        """
        self.mlp.fit(X,y,Val)
        
    def predict(self,X):
        return self.mlp.predict(X)


    def get_latent(self, X_val):
        
        z_values= self.mlp._forward_propagation(X_val) #value/ before applying activation /after forward prop in layer
        latent= z_values[ len(self.before)  ] #latent before activation/ as reluu makes -ve 0 

        return latent , self.mlp._activate( z_values[-1])#, last layer after activation
        

