import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Autoencoder class with customizable parameters
class Autoencoder(nn.Module):
    def __init__(self, latent_dim, learning_rate=1e-3, optimizer_type='Adam', kernel_size=3, num_filters=[32, 64]):
        super(Autoencoder, self).__init__()
        
        # Encoder with customizable kernel size and number of filters
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_filters[0], kernel_size=kernel_size, stride=2, padding=1),  # (batch_size, 1, 28, 28) -> (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=kernel_size, stride=2, padding=1),  # (batch_size, 32, 14, 14) -> (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output to (batch_size, 64*7*7)
            nn.Linear(num_filters[1] * 7 * 7, latent_dim),  # Latent space (batch_size, latent_dim)
            nn.ReLU()
        )
        
        # Decoder with customizable kernel size and number of filters
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, num_filters[1] * 7 * 7),  # Latent space -> (batch_size, 64*7*7)
            nn.ReLU(),
            nn.Unflatten(1, (num_filters[1], 7, 7)),  # Reshape to (batch_size, 64, 7, 7)
            nn.ConvTranspose2d(num_filters[1], num_filters[0], kernel_size=kernel_size, stride=2, padding=1, output_padding=1),  # (batch_size, 64, 7, 7) -> (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters[0], 1, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 14, 14) -> (batch_size, 1, 28, 28)
            nn.Sigmoid()  # To ensure the output is in the range [0, 1]
        )
        
        # Optimizer setup based on choice
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError("Optimizer not supported. Choose either 'Adam' or 'SGD'")
        
        self.to(device)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed
    
    # Training method
    def fit(self, train_loader, val_loader, criterion, num_epochs=1):
        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            running_loss = 0.0
            for data in train_loader:
                img, _ = data
                img = img.to(device)  # Move data to the GPU or CPU based on the device
                self.optimizer.zero_grad()
                output = self(img)
                loss = criterion(output, img)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            
            # Now calculate validation loss
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                for data in val_loader:
                    img, _ = data
                    img = img.to(device)  # Move data to the GPU or CPU based on the device
                    output = self(img)
                    loss = criterion(output, img)
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
            
            # Print the average loss for both training and validation
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    
    # Prediction method (for testing or inference)
    def predict(self, test_loader):
        self.eval()
        with torch.no_grad():
            all_reconstructed = []
            for data in test_loader:
                img, _ = data
                img = img.to(device)  # Move data to the GPU or CPU based on the device
                reconstructed = self(img)
                all_reconstructed.append(reconstructed)
            return torch.cat(all_reconstructed, dim=0)

