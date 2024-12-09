import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelCNN(nn.Module):
    def __init__(self, num_classes=10,num_conv_layers=1, dropout_rate=0.3):
        super(MultiLabelCNN, self).__init__()

        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate

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
        self.fc2 = nn.Linear(128, 10)  # Output for 10 digits (0-9)

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
    
        
        return x



import torch.optim as optim
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, optimizer='adam', device='cuda'):
    """
    Train the model using PyTorch's native training loop, with validation.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for multi-label classification

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
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()  # Ensure labels are float for BCEWithLogitsLoss
            
            optimizer.zero_grad()
            outputs = model(images)

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
                images, labels = images.to(device), labels.to(device).float()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Print the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        val_losses.append(avg_val_loss)  

    return train_losses, val_losses  # Return lists of training and validation losses

def predict(model, data_loader, device='cuda', threshold=0.5):
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
            
            # Apply sigmoid activation and threshold for multi-label classification
            outputs = torch.sigmoid(outputs)
            predicted = (outputs >= threshold).int()  # Convert to binary predictions
            predictions.extend(predicted.cpu().numpy())
    
    return predictions