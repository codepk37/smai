
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self, task='classification', num_classes=10, num_conv_layers=1, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.task = task
        self.num_conv_layers = num_conv_layers  # New parameter to specify number of conv layers, should be >2
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.extra_convs = nn.ModuleList()
        for i in range(0, num_conv_layers ): 
            self.extra_convs.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        

        # Final convolutional layer to increase channels
        self.conv_last = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes if task == 'classification' else 1)

    def forward(self, x):
        # First conv layer followed by pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Pass through each of the additional convolutional layers
        for conv in self.extra_convs:
            x = F.relu(conv(x))
        
        # Final convolutional layer
        x = self.pool(F.relu(self.conv_last(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first fully connected layer
        x = self.fc2(x)
        
        # Output layer
        if self.task == 'classification':
            return F.log_softmax(x, dim=1)
        return x
    
def train_model(model, train_loader,val_loader, num_epochs=10, learning_rate=0.001,optimizer='adam', device='cuda'):
    """
    Train the model using PyTorch's native training loop, with validation.
    """
    model = model.to(device)
    print("Check hthiss  ",model.task)
    criterion = nn.NLLLoss() if model.task == 'classification' else nn.MSELoss() #nnloss expects class indices no one hot encoding
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Choose optimizer
    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer not supported. Choose from 'adam', 'sgd', or 'rmsprop'.")
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if model.task == 'regression':
                labels = labels.float()  # Ensure labels are float for regression
            
            optimizer.zero_grad()
            outputs = model(images)

            if model.task == 'classification':
                labels = labels.long() 

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Print the average loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if model.task == 'regression':
                    labels = labels.float()
                
                outputs = model(images)

                if model.task == 'classification':
                    labels = labels.long() 

                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Print the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        val_losses.append(avg_val_loss)  

    return train_losses, val_losses  # Return lists of training and validation losses

def predict(model, data_loader, device='cuda'):
    """
    Make predictions using the trained model.
    """
    model = model.to(device)
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Check if batch has both images and labels
            if isinstance(batch, (list, tuple)):
                images = batch[0]  # Extract images only
            else:
                images = batch  # Handle case where only images are provided
            
            images = images.to(device)
            outputs = model(images)
            
            if model.task == 'classification':
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
            else:
                predictions.extend(outputs.cpu().numpy().flatten()) 
    
    return predictions



### JUST FOR VISUALIZATION, EXACTLY SAME AS ABOVE CLASS

import matplotlib.pyplot as plt

class CNNViz(nn.Module):
    def __init__(self, task='classification', num_classes=10, num_conv_layers=1, dropout_rate=0.3):
        super(CNNViz, self).__init__()
        self.task = task
        self.num_conv_layers = num_conv_layers
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.extra_convs = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.extra_convs.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        
        self.conv_last = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes if task == 'classification' else 1)

    def forward(self, x, return_feature_maps=False):
        feature_maps = []

        # First conv layer followed by pooling
        x = self.pool(F.relu(self.conv1(x)))
        feature_maps.append(x.clone())  # Store the first feature map

        # Pass through each of the additional convolutional layers
        for conv in self.extra_convs:
            x = F.relu(conv(x))
            feature_maps.append(x.clone())  # Store feature map after each layer
        
        # Final convolutional layer
        x = self.pool(F.relu(self.conv_last(x)))
        feature_maps.append(x.clone())  # Store the final feature map

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Output layer
        if self.task == 'classification':
            return F.log_softmax(x, dim=1), feature_maps if return_feature_maps else F.log_softmax(x, dim=1)
        return x, feature_maps if return_feature_maps else x

def train_modelViz(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, optimizer='adam', device='cuda'):
    model = model.to(device)
    criterion = nn.NLLLoss() if model.task == 'classification' else nn.MSELoss()
    
    # Choose optimizer
    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer not supported. Choose from 'adam', 'sgd', or 'rmsprop'.")
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images, return_feature_maps=True)  # Ensure to get feature maps
            
            # Ensure labels are of type Long for classification
            if model.task == 'classification':
                labels = labels.long()  # Convert to Long Tensor for classification
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images, return_feature_maps=True)  # Ensure to get feature maps
                
                # Ensure labels are of type Long for classification
                if model.task == 'classification':
                    labels = labels.long()  # Convert to Long Tensor for classification
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        val_losses.append(avg_val_loss)

        # Capture feature maps for visualization during the last epoch
        # Capture feature maps for visualization during the last epoch
        if epoch == num_epochs - 1:
            # Visualize feature maps for the first 3 images in the first batch
            with torch.no_grad():
                images, _ = next(iter(train_loader))
                images = images[:3].to(device)
                _, feature_maps_list = model(images, return_feature_maps=True)

                for img_idx, feature_maps in enumerate(feature_maps_list):
                    print(f"\nFeature Maps for Selected Image {img_idx + 1}:")
                    plt.figure(figsize=(15, 15))  # Adjust figure size as needed

                    for layer_idx, fmap in enumerate(feature_maps):
                        # Check the dimensions of the feature map
                        print(f"Layer {layer_idx + 1} feature map shape: {fmap.shape}")

                        # If the feature map has 3 dimensions (num_channels, height, width)
                        if fmap.dim() == 3:  
                            # Select a specific channel to visualize; you can adjust the channel index here
                            num_channels = fmap.size(0)  # Number of channels
                            for c in range(num_channels):
                                plt.subplot(num_channels, len(feature_maps), layer_idx * num_channels + c + 1)
                                fmap_channel = fmap[c, :, :].detach().cpu().numpy()  # Use detach() here
                                plt.imshow(fmap_channel, cmap='viridis')
                                plt.title(f'Layer {layer_idx + 1}, Channel {c + 1}')
                                plt.axis('off')

                        else:
                            # If the dimensions are unexpected
                            print(f"Unexpected shape for layer {layer_idx + 1}: {fmap.shape}")

                    plt.show()


    return train_losses, val_losses