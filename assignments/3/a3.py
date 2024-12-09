# For single-label classification:
# mlp = MLP(input_size=11, hidden_layers=[32, 16], output_size=5,  # Example 5-class classification
#           output_activation='softmax', loss='cross_entropy', 
#           learning_rate=0.01, epochs=100)

# mlp.fit(X, y_one_hot)  # where y_one_hot is one-hot encoded labels


# For multi-label classification:
# mlp = MLP(input_size=11, hidden_layers=[32, 16], output_size=5,  # Example 5 labels for multi-label
#           output_activation='sigmoid', loss='binary_cross_entropy', 
#           learning_rate=0.01, epochs=100)

# mlp.fit(X, y_multi_label)  # where y_multi_label contains multi-label ground truth

#2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('./data/external/WineQT.csv')

def task2_1():
    """
    Describe the dataset using mean, standard deviation, min, and max values for all attributes.
    """
    description = df.describe().T  # Transpose for better readability
    print("Dataset Description:")
    print(description[['mean', 'std', 'min', 'max']])


    """
    Draw a graph that shows the distribution of the various labels across the entire dataset.
    """
    # List of features to plot
    features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality'
    ]

    # Set the figure size
    plt.figure(figsize=(20, 15))

    # Create subplots for each feature
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 3, i)  # Create a grid of 4 rows and 3 columns
        df[feature].plot(kind='hist', bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('./assignments/3/figures/task2_2.png')
    # plt.show()  # Uncomment to display the plot


    """
    Normalize and standardize the data while handling missing or inconsistent data values if necessary.
    """
    # Check for missing values
    print("Missing values per column:\n", df.isnull().sum())

    # Fill missing values if any (example: filling with mean)
    df.fillna(df.mean(), inplace=True)

    # Separate features and labels
    X = df.drop(['quality', 'Id'], axis=1)  # Features (excluding 'quality' and 'Id')
    y = df['quality']  # Labels

    # Standardize the data (zero mean, unit variance)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Normalize the data (scales between 0 and 1)
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    # Convert normalized and standardized data into DataFrames for further use
    df_standardized = pd.DataFrame(X_standardized, columns=X.columns)
    df_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # Add the quality column back to the DataFrames
    df_standardized['quality'] = y.values
    df_normalized['quality'] = y.values

    # Save the normalized and standardized data to CSV files
    df_normalized.to_csv('./data/interim/3/WineQT_norm.csv', index=False)
    df_standardized.to_csv('./data/interim/3/WineQT_standar.csv', index=False)

    print("Data Normalization and Standardization complete. Files saved:")
    print("./data/interim/3/WineQT_norm.csv")
    print("./data/interim/3/WineQT_standar.csv")

# task2_1()


#######LOAD
df = pd.read_csv('./data/interim/3/WineQT_standar.csv')

# Separate features and labels
X = df.drop('quality', axis=1).values  # Features (11 attributes)
y = df['quality'].values  # Labels (Wine Quality)

# Check unique labels in y
print("Unique labels in y:", np.unique(y))

# Convert labels to one-hot encoding (assuming labels range from 3 to 8)
num_classes = len(np.unique(y))  # Number of unique classes
y_one_hot = np.eye(num_classes)[y - 3]  # One-hot encoding, assuming labels are [3, 4, 5, 6, 7, 8]
# (1143, 6)




np.random.seed(42)  # Set seed for reproducibility
indices = np.random.permutation(len(X))

# Define sizes for the splits
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
test_size = len(X) - train_size - val_size

# Split the indices into training, validation, and test
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Create the train, validation, and test sets
train_x, train_y = X[train_indices], y_one_hot[train_indices]
val_x, val_y = X[val_indices], y_one_hot[val_indices]
test_x, test_y = X[test_indices], y_one_hot[test_indices]
#######

from models.MLP.MLP import *

def task2_2():
    # Load the dataset


    # Initialize the MLP model
    input_size = train_x.shape[1]  # Number of features (11) 
    output_size = num_classes  # Number of classes (6 in this case: 3, 4, 5, 6, 7, 8)

    # Define hidden layers (example: two hidden layers with 32 and 16 neurons)
    hidden_layers = [32, 64 ,64,32 ]

    # Create an MLP instance for classification
    mlp = MLP(input_size=input_size, 
            hidden_layers=hidden_layers, 
            output_size=output_size,
            output_activation='softmax',  # Use softmax for multi-class classification
            loss='cross_entropy',  # Use cross-entropy loss
            learning_rate=0.01, 
            epochs=100,
            activation='relu',
            batch_size=10,
            optimizer="mini-batch" 
            )

    # Train the model
    mlp.fit(train_x, train_y)  # Use the one-hot encoded labels directly


    check_gradient =mlp.gradient_checking(train_x,train_y)
    print("Gradient check ",check_gradient)

    # Make predictions on the training data
    predictions = mlp.predict(val_x)  # Get the predicted probabilities
    predicted_indices = np.argmax(predictions, axis=1)  # For single-label classification
    actual_indices   = np.argmax(val_y, axis=1)
    # Calculate accuracy (compare predictions with original labels)
    # predicted_classes = predictions + 3  # Adjust class indices back to original labels
    accuracy = np.mean(predicted_indices == actual_indices)  # Compare with original labels
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    loss = mlp.compute_loss(val_y,predictions) #after cross entropy
    print("Validation loss ",loss) 

    test_predictions = mlp.predict(test_x)  # Get the predicted probabilities
    test_predicted_indices = np.argmax(test_predictions, axis=1)  # For single-label classification
    test_actual_indices = np.argmax(test_y, axis=1)  # Convert one-hot encoded labels back to class indices

    # Calculate test accuracy
    test_accuracy = np.mean(test_predicted_indices == test_actual_indices)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Calculate test loss
    test_loss = mlp.compute_loss(test_y, test_predictions)
    print("Test Loss:", test_loss)

# task2_2()


import wandb




import numpy as np
import pandas as pd
import wandb
from performance_measures.performance import *



def task2_3():
    # Define the sweep configuration for hyperparameter tuning
    sweep_config = {
        'method': 'grid',  # Grid search
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.001, 0.01]
            },
            'epochs': {
                'values': [100, 200]
            },
            'hidden_layers': {
                'values': [[32, 32], [32, 64, 64, 32]]
            },
            'activation': {
                'values': ['relu', 'tanh', 'sigmoid', 'linear']
            },
            'optimizer': {
                'values': ['mini-batch', 'sgd', 'batch']
            },
        }
    }
    best_config = {}
    best_accuracy = 0   
    all_metrics = []  # To store all hyperparameters and corresponding metrics

    # Define the function that will run for each configuration
    def task2_3_sweep():
        with wandb.init() as run:
            # Get the current hyperparameters from W&B sweep config
            config = wandb.config
            lr = config.learning_rate
            epoch = config.epochs
            hidden_layer = config.hidden_layers
            activation = config.activation
            optimizer = config.optimizer

            # Load dataset and split it into training, validation, and test sets
            df = pd.read_csv('./data/interim/3/WineQT_standar.csv')
            X = df.drop('quality', axis=1).values
            y = df['quality'].values
            y_one_hot = np.eye(len(np.unique(y)))[y - 3]

            np.random.seed(42)
            indices = np.random.permutation(len(X))
            train_size = int(0.8 * len(X))
            val_size = int(0.1 * len(X))

            # Split indices
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            train_x, train_y = X[train_indices], y_one_hot[train_indices]
            val_x, val_y = X[val_indices], y_one_hot[val_indices]
            test_x, test_y = X[test_indices], y_one_hot[test_indices]

            # Initialize MLP model using the current sweep configuration
            input_size = train_x.shape[1]
            output_size = len(np.unique(y))  # Number of unique classes

            mlp = MLP(input_size=input_size, 
                      hidden_layers=hidden_layer, 
                      output_size=output_size,
                      output_activation='softmax',  
                      loss='cross_entropy',  
                      learning_rate=lr, 
                      epochs=epoch,
                      activation=activation,
                      batch_size=10,
                      optimizer=optimizer)

            # Train the model
            mlp.fit(train_x, train_y)

            # Validate the model
            val_predictions = mlp.predict(val_x)
            val_predicted_indices = np.argmax(val_predictions, axis=1)
            val_actual_indices = np.argmax(val_y, axis=1)

            # Test metrics
            # Calculate validation metrics using performance_metrices
            val_metrics = performance_metrices(val_actual_indices, val_predicted_indices)

            val_accuracy = compute_accuracy(val_actual_indices, val_predicted_indices)
            val_loss = mlp.compute_loss(val_y, val_predictions)
            val_f1 = val_metrics['macro_f1']  # Use f1 from performance_metrices
            val_precision = val_metrics['macro_precision']  # Use precision from performance_metrices
            val_recall = val_metrics['macro_recall']  # Use recall from performance_metrices
    

            # Log validation metrics to W&B
            wandb.log({
                "learning_rate": lr,
                "epochs": epoch,
                "hidden_layers": hidden_layer,
                "activation": activation,
                "optimizer": optimizer,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "validation_f1_score": val_f1,
                "validation_precision": val_precision,
                "validation_recall": val_recall
            })

             # Save hyperparameters and metrics for the table
            all_metrics.append({
                "learning_rate": lr,
                "epochs": epoch,
                "hidden_layers": hidden_layer,
                "activation": activation,
                "optimizer": optimizer,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "f1_score": val_f1,
                "precision": val_precision,
                "recall": val_recall
            })

            # Track the best model
            nonlocal best_accuracy, best_config
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_config = {
                    "learning_rate": lr,
                    "epochs": epoch,
                    "hidden_layers": hidden_layer,
                    "activation": activation,
                    "optimizer": optimizer,
                    "best_f1": val_f1,
                    "best_precision": val_precision,
                    "best_recall": val_recall
                }




    # Initialize the W&B sweep and run the agent
    sweep_id = wandb.sweep(sweep_config, project="wine-quality-prediction")
    wandb.agent(sweep_id, function=task2_3_sweep)


    print("#####Best Model Configuration:") ###2.3.2  (5 marks)
    for key, value in best_config.items():
        print(f"{key}: {value}")

    # Generate a table listing all hyperparameters tested and their metrics
    df_metrics = pd.DataFrame(all_metrics)
    print("\nHyperparameter Tuning Results:")
    print(df_metrics)

    # Optionally, save the table to a CSV for reporting
    df_metrics.to_csv('./assignments/3/hyperparamerter_tuning/hyperparameter_tuning_results.csv', index=False) #(2.3.2 - 5 marks)

# https://wandb.ai/kk408-aissms-ioit/wine-quality-prediction
# Run the task to execute the hyperparameter tuning with W&B
#BEST VAL ACCURACY:  relu 100 [32,32] 0.01 sgd
# task2_3()









################################################

import numpy as np
import pandas as pd
import wandb
from performance_measures.performance import *

def task2_3_1():
    # Initialize W&B project
    wandb.init(project="wine-quality-prediction-activation-curves")

    # Load dataset
    df = pd.read_csv('./data/interim/3/WineQT_standar.csv')
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    y_one_hot = np.eye(len(np.unique(y)))[y - 3]

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_x, train_y = X[train_indices], y_one_hot[train_indices]
    val_x, val_y = X[val_indices], y_one_hot[val_indices]
    test_x, test_y = X[test_indices], y_one_hot[test_indices]

    # Fixed hyperparameters
    lr = 0.01
    optimizer = 'sgd'
    hidden_layers = [32, 32]  # You can modify this if needed
    epochs_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,190,200,210,220,230,240,250]
    activation_functions = ['relu', 'sigmoid', 'tanh', 'linear']

    for activation in activation_functions:
        # Create a new run for each activation function
        with wandb.init(project="wine-quality-prediction-activation-curves", name=f"{activation}_activation", reinit=True):
            # Data for plotting
            epochs_data = []
            train_accuracies = []
            val_accuracies = []
            train_losses = []
            val_losses = []

            for epochs in epochs_list:
                # Initialize the MLP model
                mlp = MLP(input_size=train_x.shape[1],
                          hidden_layers=hidden_layers,
                          output_size=len(np.unique(y)),
                          output_activation='softmax',
                          loss='cross_entropy',
                          learning_rate=lr,
                          epochs=epochs,
                          activation=activation,
                          batch_size=10,
                          optimizer=optimizer)

                # Train the model
                mlp.fit(train_x, train_y)
                train_predictions = mlp.predict(train_x)
                train_predicted_indices = np.argmax(train_predictions, axis=1)
                train_actual_indices = np.argmax(train_y, axis=1)
                train_accuracy = compute_accuracy(train_actual_indices, train_predicted_indices)

                val_predictions = mlp.predict(val_x)
                val_predicted_indices = np.argmax(val_predictions, axis=1)
                val_actual_indices = np.argmax(val_y, axis=1)
                val_accuracy = compute_accuracy(val_actual_indices, val_predicted_indices)

                # Get final training and validation metrics
                val_loss = mlp.compute_loss(val_y, val_predictions)
                train_loss =mlp.compute_loss(train_y, train_predictions)


                # Store data for plotting
                epochs_data.append(epochs)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics to W&B
                wandb.log({
                    "epochs": epochs,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "activation": activation
                })

            # Create line plots for accuracy and loss
            wandb.log({
                f"{activation}_accuracy_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_accuracies, val_accuracies],
                    keys=["Train Accuracy", "Validation Accuracy"],
                    title=f"{activation.capitalize()} Activation: Accuracy vs Epochs",
                    xname="Epochs"
                ),
                f"{activation}_loss_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_losses, val_losses],
                    keys=["Train Loss", "Validation Loss"],
                    title=f"{activation.capitalize()} Activation: Loss vs Epochs",
                    xname="Epochs"
                )
            })

    # Finish W&B run
    wandb.finish()

# https://wandb.ai/kk408-aissms-ioit/wine-quality-prediction-activation-curves  ;DONE
# Run the task to execute the hyperparameter tuning with W&B
# task2_3_1()




import numpy as np
import pandas as pd
import wandb
from performance_measures.performance import *

def task2_3_2():
    # Initialize W&B project
    wandb.init(project="wine-quality-prediction-optimizer-curves")

    # Load dataset
    df = pd.read_csv('./data/interim/3/WineQT_standar.csv')
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    y_one_hot = np.eye(len(np.unique(y)))[y - 3]

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_x, train_y = X[train_indices], y_one_hot[train_indices]
    val_x, val_y = X[val_indices], y_one_hot[val_indices]
    test_x, test_y = X[test_indices], y_one_hot[test_indices]

    # Fixed hyperparameters
    lr = 0.01
    hidden_layers = [32, 32]  # You can modify this if needed
    epochs_list = [10,20,30,40,50,60,70,80,90,100]  # List of epochs to evaluate
    optimizers = ['sgd', 'mini-batch', 'batch']  # Optimizers to test
    activation_function = 'relu'  # Fixed activation function

    for optimizer in optimizers:
        # Create a new run for each optimizer
        with wandb.init(project="wine-quality-prediction-optimizer-curves", name=f"{optimizer}_optimizer", reinit=True):
            # Data for plotting
            epochs_data = []
            train_accuracies = []
            val_accuracies = []
            train_losses = []
            val_losses = []

            for epochs in epochs_list:
                # Initialize the MLP model
                mlp = MLP(input_size=train_x.shape[1],
                          hidden_layers=hidden_layers,
                          output_size=len(np.unique(y)),
                          output_activation='softmax',
                          loss='cross_entropy',
                          learning_rate=lr,
                          epochs=epochs,
                          activation=activation_function,
                          batch_size=10,
                          optimizer=optimizer)

                # Train the model
                mlp.fit(train_x, train_y)

                # Validate the model
                val_predictions = mlp.predict(val_x)
                val_predicted_indices = np.argmax(val_predictions, axis=1)
                val_actual_indices = np.argmax(val_y, axis=1)

                train_predictions = mlp.predict(train_x)
                train_predicted_indices = np.argmax(train_predictions, axis=1)
                train_actual_indices = np.argmax(train_y, axis=1)

                # Calculate metrics
                train_accuracy = compute_accuracy(train_actual_indices, train_predicted_indices)
                val_accuracy = compute_accuracy(val_actual_indices, val_predicted_indices)
                train_loss = mlp.compute_loss(train_y, train_predictions)
                val_loss = mlp.compute_loss(val_y, val_predictions)

                # Store data for plotting
                epochs_data.append(epochs)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics to W&B
                wandb.log({
                    "epochs": epochs,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "optimizer": optimizer
                })

            # Create line plots for accuracy and loss
            wandb.log({
                f"{optimizer}_accuracy_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_accuracies, val_accuracies],
                    keys=["Train Accuracy", "Validation Accuracy"],
                    title=f"{optimizer.capitalize()} Optimizer: Accuracy vs Epochs",
                    xname="Epochs"
                ),
                f"{optimizer}_loss_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_losses, val_losses],
                    keys=["Train Loss", "Validation Loss"],
                    title=f"{optimizer.capitalize()} Optimizer: Loss vs Epochs",
                    xname="Epochs"
                )
            })

    # Finish W&B run
    wandb.finish()

# https://wandb.ai/kk408-aissms-ioit/wine-quality-prediction-optimizer-curves
# Run the task to execute the hyperparameter tuning with W&B
# task2_3_2()


import numpy as np
import pandas as pd
import wandb
from performance_measures.performance import *

def task2_3_3():
    # Initialize W&B project
    wandb.init(project="wine-quality-prediction-learning-rate-curves")

    # Fixed hyperparameters
    hidden_layers = [32, 32]  # You can modify this if needed
    epochs_list = [10,20,30,40,50,60, 70,80,90,100,110,120]  # List of epochs to evaluate
    learning_rates = [0.01, 0.03, 0.07, 0.10]  # Learning rates to test
    activation_function = 'relu'  # Fixed activation function

    for lr in learning_rates:
        # Create a new run for each learning rate
        with wandb.init(project="wine-quality-prediction-learning-rate-curves", name=f"lr_{lr}", reinit=True):
            # Data for plotting
            epochs_data = []
            train_accuracies = []
            val_accuracies = []
            train_losses = []
            val_losses = []

            for epochs in epochs_list:
                # Initialize the MLP model
                mlp = MLP(input_size=train_x.shape[1],
                          hidden_layers=hidden_layers,
                          output_size=len(np.unique(y)),
                          output_activation='softmax',
                          loss='cross_entropy',
                          learning_rate=lr,
                          epochs=epochs,
                          activation=activation_function,
                          batch_size=10,
                          optimizer='sgd')  # You can modify this if needed

                # Train the model
                mlp.fit(train_x, train_y)

                # Validate the model
                val_predictions = mlp.predict(val_x)
                val_predicted_indices = np.argmax(val_predictions, axis=1)
                val_actual_indices = np.argmax(val_y, axis=1)

                train_predictions = mlp.predict(train_x)
                train_predicted_indices = np.argmax(train_predictions, axis=1)
                train_actual_indices = np.argmax(train_y, axis=1)

                # Calculate metrics
                train_accuracy = compute_accuracy(train_actual_indices, train_predicted_indices)
                val_accuracy = compute_accuracy(val_actual_indices, val_predicted_indices)
                train_loss = mlp.compute_loss(train_y, train_predictions)
                val_loss = mlp.compute_loss(val_y, val_predictions)

                # Store data for plotting
                epochs_data.append(epochs)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics to W&B
                wandb.log({
                    "epochs": epochs,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "learning_rate": lr
                })

            # Create line plots for accuracy and loss
            wandb.log({
                f"lr_{lr}_accuracy_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_accuracies, val_accuracies],
                    keys=["Train Accuracy", "Validation Accuracy"],
                    title=f"Learning Rate: {lr} - Accuracy vs Epochs",
                    xname="Epochs"
                ),
                f"lr_{lr}_loss_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_losses, val_losses],
                    keys=["Train Loss", "Validation Loss"],
                    title=f"Learning Rate: {lr} - Loss vs Epochs",
                    xname="Epochs"
                )
            })

    # Finish W&B run
    wandb.finish()
# https://wandb.ai/kk408-aissms-ioit/wine-quality-prediction-learning-rate-curves
# Run the task to execute the hyperparameter tuning with W&B
# task2_3_3()



######################

# best val accuracy at :relu 100 [32,32] 0.01 sgd


from performance_measures.performance import *

def task2_4():

    # Use the best model hyperparameters from tuning
    best_lr = 0.01  # Replace with the tuned best learning rate
    best_hidden_layers = [32, 32]  # Replace with the best number of layers/nodes
    best_epochs = 100  # Replace with the best number of epochs
    best_activation = 'relu'  # Replace with the best activation function

    # Initialize the best MLP model
    best_mlp = MLP(input_size=train_x.shape[1],
                   hidden_layers=best_hidden_layers,
                   output_size=len(np.unique(y)),
                   output_activation='softmax',
                   loss='cross_entropy',
                   learning_rate=best_lr,
                   epochs=best_epochs,
                   activation=best_activation,
                   batch_size=10,
                   optimizer='sgd')

    # Train the best model on training data
    best_mlp.fit(train_x, train_y)

    # Evaluate on test set
    test_predictions = best_mlp.predict(test_x)
    test_predicted_indices = np.argmax(test_predictions, axis=1)
    test_actual_indices = np.argmax(test_y, axis=1)

    # Compute accuracy
    test_accuracy = compute_accuracy(test_actual_indices, test_predicted_indices)

    test_metrics = performance_metrices(test_actual_indices, test_predicted_indices)

    # Compute precision, recall, F1-score
    precision =  test_metrics['macro_precision'] 
    recall =   test_metrics['macro_recall'] 
    f1_score = test_metrics['macro_f1'] 

    # Compute loss on test data
    test_loss = best_mlp.compute_loss(test_y, test_predictions)


    # Print metrics for reporting
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test F1-Score: {f1_score}")
    print(f"Test Loss: {test_loss}")

# Test Accuracy: 0.5304347826086957
# Test Precision: 0.25267094017094016
# Test Recall: 0.27267864783910994
# Test F1-Score: 0.25496031746031744
# Test Loss: 4.361407765410269
# task2_4()






##############################
#5)  Analyze part: Report remaining


# 1. Effect of Non-linearity: #call task2_3_1
def task2_5_1():
    task2_3_1()  #same function up implemented
# task2_5_1()
# TAKE BELOW LINK RESULTS ;which used best hyperparameter+changing_activaion 
# https://wandb.ai/kk408-aissms-ioit/wine-quality-prediction-activation-curves 




# 2. Effect of Learning Rate:
def task2_5_2():
    task2_3_3()  # same function up implemented
# task2_5_2()
# TAKE BELOW LINK RESULTS ;which used best hyperparameter+changing_learning rate
# https://wandb.ai/kk408-aissms-ioit/wine-quality-prediction-learning-rate-curves


# 3. Effect of Batch Size
def task2_5_3():
    # Initialize W&B project
    wandb.init(project="wine-quality-prediction-batch-size-curves")

    
    # Fixed hyperparameters
    lr = 0.01
    activation_function = 'relu'  # Fixed activation function
    optimizer = 'mini-batch'
    hidden_layers = [32, 32]  # You can modify this if needed
    epochs_list = [10,20,30,40]#,50,60,70,80,90,100,110, 120]  # List of epochs to evaluate
    batch_sizes = [10, 20, 50, 100]  # Batch sizes to test

    for batch_size in batch_sizes:
        # Create a new run for each batch size
        with wandb.init(project="wine-quality-prediction-batch-size-curves", name=f"batch_size_{batch_size}", reinit=True):
            # Data for plotting
            epochs_data = []
            train_accuracies = []
            val_accuracies = []
            train_losses = []
            val_losses = []

            for epochs in epochs_list:
                # Initialize the MLP model
                mlp = MLP(input_size=train_x.shape[1],
                          hidden_layers=hidden_layers,
                          output_size=len(np.unique(y)),
                          output_activation='softmax',
                          loss='cross_entropy',
                          learning_rate=lr,
                          epochs=epochs,
                          activation=activation_function,
                          batch_size=batch_size,
                          optimizer=optimizer)

                # Train the model
                mlp.fit(train_x, train_y)

                # Validate the model
                val_predictions = mlp.predict(val_x)
                val_predicted_indices = np.argmax(val_predictions, axis=1)
                val_actual_indices = np.argmax(val_y, axis=1)

                train_predictions = mlp.predict(train_x)
                train_predicted_indices = np.argmax(train_predictions, axis=1)
                train_actual_indices = np.argmax(train_y, axis=1)

                # Calculate metrics
                train_accuracy = compute_accuracy(train_actual_indices, train_predicted_indices)
                val_accuracy = compute_accuracy(val_actual_indices, val_predicted_indices)
                train_loss = mlp.compute_loss(train_y, train_predictions)
                val_loss = mlp.compute_loss(val_y, val_predictions)

                # Store data for plotting
                epochs_data.append(epochs)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics to W&B
                wandb.log({
                    "epochs": epochs,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "batch_size": batch_size
                })

            # Create line plots for accuracy and loss
            wandb.log({
                f"batch_size_{batch_size}_accuracy_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_accuracies, val_accuracies],
                    keys=["Train Accuracy", "Validation Accuracy"],
                    title=f"Batch Size: {batch_size} - Accuracy vs Epochs",
                    xname="Epochs"
                ),
                f"batch_size_{batch_size}_loss_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_losses, val_losses],
                    keys=["Train Loss", "Validation Loss"],
                    title=f"Batch Size: {batch_size} - Loss vs Epochs",
                    xname="Epochs"
                )
            })

    # Finish W&B run
    wandb.finish()

# https://wandb.ai/kk408-aissms-ioit/wine-quality-prediction-batch-size-curves
# Run the task to execute the hyperparameter tuning with W&B
# task2_5_3()



#########################
# 2.6 Multi-Label Classification




from sklearn.preprocessing import MultiLabelBinarizer

def preprocess():
        # Load the dataset
    advertisement = pd.read_csv('./data/external/advertisement.csv')

    # Split the 'labels' column which contains multi-label strings into lists for easier processing
    advertisement['labels'] = advertisement['labels'].str.split()

    # Map 'gender' column to numeric values (Male: 0, Female: 1)
    advertisement['gender'] = advertisement['gender'].map({'Male': 0, 'Female': 1})

    # One-hot encode the 'occupation' column to create separate binary columns for each unique occupation
    occupation_dummies = pd.get_dummies(advertisement['occupation'], prefix='occupation')
    occupation_dummies = occupation_dummies.astype(int)

    advertisement = pd.concat([advertisement, occupation_dummies], axis=1)

    # Binarize the 'labels' column for multi-label classification
    # This converts the labels into a binary matrix where each label gets its own column
    binarizer = MultiLabelBinarizer()
    vecs = binarizer.fit_transform(advertisement['labels'])
    binarized_df = pd.DataFrame(vecs, columns=binarizer.classes_)
    advertisement = pd.concat([advertisement, binarized_df], axis=1)

    # Drop unnecessary columns: 'labels' (since we binarized it), 'most bought item', 'city', and 'occupation' 
    # (as occupation is now represented with the one-hot encoded columns)
    advertisement.drop(['labels', 'most bought item', 'city', 'occupation'], axis=1, inplace=True)

    # Encode 'married' column to binary values (True: 1, False: 0)
    advertisement['married'] = advertisement['married'].astype(int)

    # Map 'education' column to numeric values: 'High School' as 1, 'Bachelor' as 2, 'Master' as 3
    advertisement['education'] = advertisement['education'].map({
        'High School': 1,
        'Bachelor': 2,
        'Master': 3,
        'PhD' :4
    })



    # Normalize the continuous numeric columns (e.g., age, income, purchase_amount)
    continuous_columns = ['age', 'income', 'purchase_amount', 'children']  # Add more if needed
    advertisement[continuous_columns] = (advertisement[continuous_columns] - advertisement[continuous_columns].mean()) / advertisement[continuous_columns].std()

    # Binary columns (e.g., gender, married, one-hot encoded occupation fields) should not be normalized.
    # No need to normalize them, as they are already in binary form (0/1)

    # Save the processed dataset to a CSV file for future use
    advertisement.to_csv('./data/interim/3/advertisement.csv', index=False)


def evaluate_metrics(y_true, y_pred):
        # Convert predictions and true labels to binary arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate instance-level accuracy
        accuracy = np.mean(np.all(y_true == y_pred, axis=1))  # Accuracy per instance

        # Initialize metrics
        true_positives = np.zeros(y_true.shape[1])  # TP for each label
        false_positives = np.zeros(y_true.shape[1])  # FP for each label
        false_negatives = np.zeros(y_true.shape[1])  # FN for each label

        # Calculate TP, FP, FN for each label
        for i in range(y_true.shape[1]):
            true_positives[i] = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            false_positives[i] = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            false_negatives[i] = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))

        # Calculate precision, recall, and F1-score for each label
        precision = true_positives / (true_positives + false_positives + 1e-8)  # Prevent division by zero
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        # Calculate macro metrics
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro + 1e-8)

        # Calculate Hamming Loss
        hamming = np.mean(y_true != y_pred)

        # Print the results
        # print(f": {accuracy * 100:.2f}%")
        # print(f"Precision: {precision_macro * 100:.2f}%")
        # print(f"Recall: {recall_macro * 100:.2f}%")
        # print(f"F1-SAccuracycore: {f1_macro * 100:.2f}%")
        # print(f"Hamming Loss: {hamming:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision_macro,
            "recall": recall_macro,
            "f1_score": f1_macro,
            "hamming_loss": hamming
        }


def task2_6_2():
    
    preprocess()

    print("Preprocessed dataset saved to './data/interim/3/advertisement.csv'")

    # Load the preprocessed dataset
    advertisement = pd.read_csv('./data/interim/3/advertisement.csv')

    # Separate the features (X) and the multi-label targets (Y)
    # Assuming the last 8 columns represent the labels, and the rest are features
    X = advertisement.iloc[:, :-8].values  # Features
    Y = advertisement.iloc[:, -8:].values  # Labels (multi-labels, one for each category)


    # Set the random seed for reproducibility
    np.random.seed(42)

    # Shuffle the indices
    indices = np.random.permutation(len(X))  # Shuffle indices directly

    # Define the split sizes
    test_size = int(0.1 * len(X))  # 10% for test set
    val_size = int(0.2 * (len(X) - test_size))  # 20% of the remaining for validation set

    # Create the train, validation, and test sets using slicing
    train_indices = indices[test_size + val_size:]  # Remaining for training set
    val_indices = indices[test_size:test_size + val_size]  # Next portion for validation set
    test_indices = indices[:test_size]  # First portion for test set

    # Create the train, validation, and test sets
    train_x, train_y = X[train_indices], Y[train_indices]
    val_x, val_y = X[val_indices], Y[val_indices]
    test_x, test_y = X[test_indices], Y[test_indices]

    # Define the model: Multi-label MLP
    input_size = train_x.shape[1]   # Number of features
    hidden_layers = [64 ,64]        # Example hidden layers
    output_size = train_y.shape[1]  # Number of labels (multi-label output)

    # Initialize the MLP for multi-label classification
    mlp = MultiLabelMLP(input_size=input_size,
                        hidden_layers=hidden_layers,
                        output_size=output_size,
                        learning_rate=0.01,
                        activation='relu',
                        output_activation='sigmoid',
                        loss='binary_cross_entropy',
                        batch_size=100,
                        epochs=200)

    # Train the model
    mlp.fit(train_x, train_y)


    # Evaluate the model on the test set
    predictions = mlp.predict(test_x)

    # Calculate accuracy (for multi-label classification, you can calculate the accuracy per label or use a metric like F1-score)
    accuracy = np.mean((predictions == test_y).astype(int))  # Accuracy metric
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Epoch 26340: Loss = 0.3768

    metrics = evaluate_metrics(test_y, predictions)#prints in function
    print(metrics)


# Test Accuracy: 66.25%
# Accuracy: 3.00%
# Precision: 51.42%
# Recall: 6.11%
# F1-Score: 10.92%
# Hamming Loss: 0.3375
# task2_6_2() #UNCOMMENT



# ////////////////////////////////////////////////////

# 2 6 3
import csv
import numpy as np
import pandas as pd

# Example hyperparameter grid
hyperparameter_grid = [
    {'hidden_layers': [64, 64], 'learning_rate': 0.01, 'batch_size': 100, 'epochs': 100},
    {'hidden_layers': [64, 128, 64], 'learning_rate': 0.001, 'batch_size': 50, 'epochs': 50},
    {'hidden_layers': [64, 64], 'learning_rate': 0.01, 'batch_size': 100, 'epochs': 100},
    {'hidden_layers': [64, 128, 64], 'learning_rate': 0.001, 'batch_size': 50, 'epochs': 50},
    {'hidden_layers': [64,64], 'learning_rate': 0.01, 'batch_size': 100, 'epochs': 100},
    {'hidden_layers': [64,128, 64], 'learning_rate': 0.001, 'batch_size': 50, 'epochs': 50},
        {'hidden_layers': [64, 64], 'learning_rate': 0.01, 'batch_size': 100, 'epochs': 30},
    {'hidden_layers': [64, 128, 64], 'learning_rate': 0.001, 'batch_size': 50, 'epochs': 80},
    {'hidden_layers': [64, 64], 'learning_rate': 0.01, 'batch_size': 100, 'epochs': 30},
    {'hidden_layers': [64, 128, 64], 'learning_rate': 0.001, 'batch_size': 50, 'epochs': 80},
    {'hidden_layers': [64,64], 'learning_rate': 0.01, 'batch_size': 100, 'epochs': 30},
    {'hidden_layers': [64,128, 64], 'learning_rate': 0.001, 'batch_size': 50, 'epochs': 80},
    # Add more hyperparameter configurations as needed
]

def task2_6_3():
    # Preprocess dataset
    preprocess()

    print("Preprocessed dataset saved to './data/interim/3/advertisement.csv'")

    # Load the preprocessed dataset
    advertisement = pd.read_csv('./data/interim/3/advertisement.csv')

    # Separate features (X) and labels (Y)
    X = advertisement.iloc[:, :-8].values  # Features
    Y = advertisement.iloc[:, -8:].values  # Labels

    # Set random seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(X))

    # Define split sizes
    test_size = int(0.1 * len(X))  # 10% for test set
    val_size = int(0.2 * (len(X) - test_size))  # 20% of remaining for validation set

    # Create train, validation, and test indices
    train_indices = indices[test_size + val_size:]
    val_indices = indices[test_size:test_size + val_size]
    test_indices = indices[:test_size]

    # Create train, validation, and test sets
    train_x, train_y = X[train_indices], Y[train_indices]
    val_x, val_y = X[val_indices], Y[val_indices]
    test_x, test_y = X[test_indices], Y[test_indices]
    
    input_size = train_x.shape[1]  # Number of features
    output_size = train_y.shape[1]  # Number of labels (multi-label output)
    # Open CSV file to store hyperparameters and metrics
    with open('./assignments/3/hyperparamerter_tuning/hyperparameter_metrics2_6.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write CSV header
        writer.writerow(['Hidden Layers', 'Learning Rate', 'Batch Size', 'Epochs', 'val_Accuracy', 'val_Precision', 'val_Recall', 'val_F1_Score', 'val_Hamming_Loss', 'train_Accuracy', 'train_Hamming_Loss'])

        for params in hyperparameter_grid:
            # Extract hyperparameters
            hidden_layers = params['hidden_layers']
            learning_rate = params['learning_rate']
            batch_size = params['batch_size']
            epochs = params['epochs']

            # Initialize MLP model
            
            mlp = MultiLabelMLP(
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=output_size,
                learning_rate=learning_rate,
                activation='relu',
                output_activation='sigmoid',
                loss='binary_cross_entropy',
                batch_size=batch_size,
                epochs=epochs
            )

            # Train the model
            mlp.fit(train_x, train_y)

            # Evaluate the model on the test set
            predictions = mlp.predict(val_x)

            # Evaluate metrics
            metrics = evaluate_metrics(val_y, predictions)

            predictions2 = mlp.predict(train_x)
            metrics2 = evaluate_metrics(train_y, predictions2)

            # Log the hyperparameters and metrics in the CSV
            writer.writerow([
                hidden_layers, 
                learning_rate, 
                batch_size, 
                epochs, 
                metrics['accuracy'], 
                metrics['precision'], 
                metrics['recall'], 
                metrics['f1_score'], 
                metrics['hamming_loss'],
                metrics2['accuracy'],
                metrics2['hamming_loss']
            ])

    #3) FOunded best hyperparameter , below ,will use best on test data
    # assignments\3\hyperparameter_metrics2_6.csv 
    # Configuration 1: Hidden Layers: [64, 128, 64], Learning Rate: 0.001, Batch Size: 50, Epochs: 50, Validation Precision: 0.4081
    # Configuration 3: Hidden Layers: [64, 64], Learning Rate: 0.01, Batch Size: 100, Epochs: 30, Validation Precision: 0.3984
    # Configuration 2: Hidden Layers: [64, 128, 64], Learning Rate: 0.001, Batch Size: 50, Epochs: 80, Validation Precision: 0.2398
    # task2_6_3()

    mlp = MultiLabelMLP(
                input_size=input_size,
                hidden_layers=[64,128,64],
                output_size=output_size,
                learning_rate=0.001,
                activation='relu',
                output_activation='sigmoid',
                loss='binary_cross_entropy',
                batch_size=50,
                epochs=50
            )

     # Train the model
    mlp.fit(train_x, train_y)

    #3) Evaluate the model on the test set
    predictions = mlp.predict(test_x)

    # Evaluate metrics
    accuracy = np.mean((predictions == test_y).astype(int))  # Accuracy metric
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    metrics = evaluate_metrics(test_y, predictions)
    print("Test set metrices ",metrics)
    # With above add this in report
    # Test Accuracy: 64.38% ; 0=0 ,1=1 mean
    #accuracy 0.01 =>1% is strict; all rows should be same [, , , ]= [ , , , ]
    # Test set metrices  {'accuracy': 0.01, 'precision': 0.3299873730188702, 'recall': 0.07140038778503166, 'f1_score': 0.11739882533303224, 'hamming_loss': 0.35625}

# task2_6_3()



def task2_7():
    preprocess()

    print("Preprocessed dataset saved to './data/interim/3/advertisement.csv'")

    # Load the preprocessed dataset
    advertisement = pd.read_csv('./data/interim/3/advertisement.csv')

    # Separate the features (X) and the multi-label targets (Y)
    # Assuming the last 8 columns represent the labels, and the rest are features
    X = advertisement.iloc[:, :-8].values  # Features
    Y = advertisement.iloc[:, -8:].values  # Labels (multi-labels, one for each category)


    # Set the random seed for reproducibility
    np.random.seed(42)

    # Shuffle the indices
    indices = np.random.permutation(len(X))  # Shuffle indices directly

    # Define the split sizes
    test_size = int(0.1 * len(X))  # 10% for test set
    val_size = int(0.2 * (len(X) - test_size))  # 20% of the remaining for validation set

    # Create the train, validation, and test sets using slicing
    train_indices = indices[test_size + val_size:]  # Remaining for training set
    val_indices = indices[test_size:test_size + val_size]  # Next portion for validation set
    test_indices = indices[:test_size]  # First portion for test set

    # Create the train, validation, and test sets
    train_x, train_y = X[train_indices], Y[train_indices]
    val_x, val_y = X[val_indices], Y[val_indices]
    test_x, test_y = X[test_indices], Y[test_indices]

    # Define the model: Multi-label MLP
    input_size = train_x.shape[1]   # Number of features
    hidden_layers = [64 ,64]        # Example hidden layers
    output_size = train_y.shape[1]  # Number of labels (multi-label output)

    # Initialize the MLP for multi-label classification
    mlp = MultiLabelMLP(input_size=input_size,
                        hidden_layers=hidden_layers,
                        output_size=output_size,
                        learning_rate=0.01,
                        activation='relu',
                        output_activation='sigmoid',
                        loss='binary_cross_entropy',
                        batch_size=100,
                        epochs=200)

    # Train the model
    mlp.fit(train_x, train_y)


    # Evaluate the model on the test set
    predictions = mlp.predict(test_x)

    # Calculate accuracy (for multi-label classification, you can calculate the accuracy per label or use a metric like F1-score)
    accuracy = np.mean((predictions == test_y).astype(int))  # Accuracy metric , considers 0=0 ,1=1
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # metrics = evaluate_metrics(test_y, predictions)#prints in function
    # print(metrics)

    correctlyclassfied =(predictions==test_y)
    print(correctlyclassfied)
    # Classes corresponding to columns
    classes = ['beauty', 'books', 'clothing', 'electronics', 'food', 'furniture', 'home', 'sports']

    # Initialize performance metrics
    performance = {cls: {'TP': 0, 'FN': 0} for cls in classes}

    # Iterate through each prediction
    for row in correctlyclassfied:
        for idx, val in enumerate(row):
            if val:  # True
                performance[classes[idx]]['TP'] += 1
            else:  # False
                performance[classes[idx]]['FN'] += 1

    # Calculate accuracy for each class
    accuracy = {cls: (perf['TP'] / (perf['TP'] + perf['FN'])) if (perf['TP'] + perf['FN']) > 0 else 0 
                for cls, perf in performance.items()}

    # Display results
    for cls, acc in accuracy.items():
        print(f"Accuracy for {cls}: {acc:.2f}")
    # Accuracy for beauty: 0.59
    # Accuracy for books: 0.69
    # Accuracy for clothing: 0.63
    # Accuracy for electronics: 0.74
    # Accuracy for food: 0.64
    # Accuracy for furniture: 0.68
    # Accuracy for home: 0.64
    # Accuracy for sports: 0.69
# task2_7()








###########################################################################################

# 3)    Multilayer Perceptron Regression

def task3_1():    


    # Load the dataset
    df = pd.read_csv('./data/external/HousingData.csv')

    # Check for NaN values and handle them by replacing with column median
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()  # Calculate median
            df[column].fillna(median_value, inplace=True)  # Replace NaN with median

    # Step 1: Describe the dataset using mean, std, min, and max
    description = pd.DataFrame({
        'mean': df.mean(),
        'std dev': df.std(),
        'min': df.min(),
        'max': df.max()
    })

    # Print the descriptive statistics
    print("Descriptive Statistics:")
    print(description)



    plt.figure(figsize=(20, 15))
    features = df.columns
    # Loop through each feature and plot its histogram
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 4, i)  # Create a grid of 4 rows and 3 columns
        df[feature].plot(kind='hist', bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

    # Adjust the layout to prevent overlap between subplots
    plt.tight_layout()

    # Show the plot
    plt.savefig("./assignments/3/figures/task3_1_distribution.png")
    # plt.show()



    # Normalization (scaling between 0 and 1)
    def normalize(data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        # Avoid division by zero by checking if max and min are not equal
        range_val = max_val - min_val
        return (data - min_val) / range_val

    # Standardization (mean=0, std=1)
    def standardize(data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero by checking if std is not zero
        std[std == 0] = 1  # Replace 0 std with 1 to avoid division by zero
        return (data - mean) / std

    # Split the dataset into features and target variable
    X = df.drop(columns=['MEDV']).values  # Features
    y = df['MEDV'].values  # Target variable

    # Normalize and standardize the feature sets
    X_normalized = normalize(X)
    X_standardized = standardize(X)

    # Print to check for NaN values after processing
    print("NaN values in normalized data:", np.isnan(X_normalized).sum())
    print("NaN values in standardized data:", np.isnan(X_standardized).sum())

    # Convert to DataFrames for further use
    df_normalized = pd.DataFrame(X_normalized, columns=df.columns[:-1])
    df_normalized['MEDV'] = y

    df_standardized = pd.DataFrame(X_standardized, columns=df.columns[:-1])
    df_standardized['MEDV'] = y

    # Save the processed datasets if needed
    df_normalized.to_csv('./data/interim/3/HousingData_normalized.csv', index=False)
    df_standardized.to_csv('./data/interim/3/HousingData_standardized.csv', index=False)

    print("Processed datasets saved.")


# task3_1()
# Done  df[column].fillna(median_value, inplace=True)  # Replace NaN with median
#                mean     std dev        min       max
# CRIM       3.479140    8.570832    0.00632   88.9762
# ZN        10.768775   23.025124    0.00000  100.0000
# INDUS     11.028893    6.704679    0.46000   27.7400
# CHAS       0.067194    0.250605    0.00000    1.0000
# NOX        0.554695    0.115878    0.38500    0.8710
# RM         6.284634    0.702617    3.56100    8.7800
# AGE       68.845850   27.486962    2.90000  100.0000
# DIS        3.795043    2.105710    1.12960   12.1265
# RAD        9.549407    8.707259    1.00000   24.0000
# TAX      408.237154  168.537116  187.00000  711.0000
# PTRATIO   18.455534    2.164946   12.60000   22.0000
# B        356.674032   91.294864    0.32000  396.9000
# LSTAT     12.664625    7.017219    1.73000   37.9700
# MEDV      22.532806    9.197104    5.00000   50.0000



def task3_2():    

    # Load the standardized dataset from a single CSV file
    df = pd.read_csv('./data/interim/3/HousingData_standardized.csv')

    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=['MEDV']).values  # Features
    y = df['MEDV'].values  # Target variable
    print("NaN values in the dataset:", np.isnan(df).sum().sum())

    # Define the split sizes
    test_size = int(0.1 * len(df))  # 10% for test
    val_size = int(0.1 * len(df))   # 10% for validation
    # Randomly shuffle the dataset
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.permutation(len(df))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]


    train_x, train_y = X[train_indices], y[train_indices]
    val_x, val_y = X[val_indices], y[val_indices]
    test_x, test_y = X[test_indices], y[test_indices]


    # Define the model parameters
    input_size = train_x.shape[1]  # Number of features
    output_size = 1  # For regression
    hidden_layers = [10,20,30,20,10]  # Example hidden layer sizes

    # Initialize the MLP regressor
    # mlp = MLPRegressor(input_size=input_size,
    #                 hidden_layers=hidden_layers,
    #                 output_size=output_size,
    #                 learning_rate=0.001,
    #                 activation='relu',
    #                 optimizer='mini-batch',
    #                 batch_size=10,
    #                 epochs=1000
    #                 )
    mlp =MLPRegressorMultiOutput(input_size=input_size,
                hidden_layers=[10,20,30,20,10],
                output_size=1,
                learning_rate=0.001,
                activation='relu',
                optimizer='mini-batch',
                batch_size=100,
                epochs=1000
                )

    # Fit the model on the training data
    print("Input contains NaN:", np.isnan(train_x).sum())
    train_y = train_y.reshape(-1, 1)
    mlp.fit(train_x, train_y)

    checkgradient =mlp.gradient_checking(train_x,train_y) ###
    print("check gradint " ,checkgradient)

    # Make predictions on the test data
    predictions = mlp.predict(test_x)

    # Compute metrics
    test_y = test_y.reshape(-1, 1)
    mse =  mlp.MSE(test_y,predictions)
    print(f'Mean Squared Error: {mse}')

    rmse = mlp.RMSE(test_y, predictions)
    print(f'Root Mean Squared Error: {rmse}')

    rsquared = mlp.Rsquared(test_y, predictions)
    print(f'R-squared: {rsquared}')

    loss =mlp.compute_loss(test_y,predictions)
    print("val  loss:",loss)  #mse

    # If needed, calculate additional metrics here (e.g., R², etc.)

# task3_2()


#########
# 3.3 Model Training & Hyperparameter Tuning using W&B
# 1) Plot the trend of loss values (MSE) with change in these hyperparameters using W&B


def task3_3_1():
    # Initialize W&B project
    df = pd.read_csv('./data/interim/3/HousingData_standardized.csv')

    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=['MEDV']).values  # Features
    y = df['MEDV'].values  # Target variable
    print("NaN values in the dataset:", np.isnan(df).sum().sum())

    # Define the split sizes
    test_size = int(0.1 * len(df))  # 10% for test
    val_size = int(0.1 * len(df))   # 10% for validation
    # Randomly shuffle the dataset
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.permutation(len(df))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]


    train_x, train_y = X[train_indices], y[train_indices]
    val_x, val_y = X[val_indices], y[val_indices]
    test_x, test_y = X[test_indices], y[test_indices]

    # Fixed hyperparameters
    lr = 0.001
    optimizer = 'sgd'
    hidden_layers = [10,20,10]  # You can modify this if needed
    epochs_list = [50,100,150,200,250,300]#, need to decide epoch
    activation_functions = ['relu', 'sigmoid', 'tanh', 'linear']

    for activation in activation_functions:
        # Create a new run for each activation function
        with wandb.init(project="HousingData-3_3-prediction-activation-curves", name=f"{activation}_activation", reinit=True):
            # Data for plotting
            epochs_data = []
            train_losses = []
            val_losses = []

            for epochs in epochs_list:
                # Initialize the MLP model
                mlp = MLPRegressorMultiOutput(input_size=train_x.shape[1],
                          hidden_layers=hidden_layers,
                          output_size=1,
                          learning_rate=lr,
                          activation=activation,
                          optimizer=optimizer,
                          batch_size=100,
                          epochs=epochs,
                          )

                # Train the model
                train_y = train_y.reshape(-1, 1)
                mlp.fit(train_x, train_y)

                train_predictions = mlp.predict(train_x)
                train_y = train_y.reshape(-1, 1)
                train_loss =  mlp.compute_loss(test_y,train_predictions)#

                val_predictions = mlp.predict(val_x)
                val_y = val_y.reshape(-1, 1)
                val_loss =  mlp.compute_loss(test_y,val_predictions)#


                # Store data for plotting
                epochs_data.append(epochs)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics to W&B
                wandb.log({
                    "epochs": epochs,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "activation": activation
                })

            # Create line plots for accuracy and loss
            wandb.log({
                f"{activation}_loss_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_losses, val_losses],
                    keys=["Train Loss", "Validation Loss"],
                    title=f"{activation.capitalize()} Activation: Loss vs Epochs",
                    xname="Epochs"
                )
            })

    # Finish W&B run
    wandb.finish()

# https://wandb.ai/kk408-aissms-ioit/HousingData-3_3-prediction-activation-curves
# task3_3_1()



def task3_3_2():
    # Initialize W&B project
    df = pd.read_csv('./data/interim/3/HousingData_standardized.csv')

    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=['MEDV']).values  # Features
    y = df['MEDV'].values  # Target variable
    print("NaN values in the dataset:", np.isnan(df).sum().sum())

    # Define the split sizes
    test_size = int(0.1 * len(df))  # 10% for test
    val_size = int(0.1 * len(df))   # 10% for validation
    # Randomly shuffle the dataset
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.permutation(len(df))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]


    train_x, train_y = X[train_indices], y[train_indices]
    val_x, val_y = X[val_indices], y[val_indices]
    test_x, test_y = X[test_indices], y[test_indices]


    # Fixed hyperparameters
    hidden_layers = [10,20,10]  # You can modify this if needed
    epochs_list = [50,100,150,200,250,300]#, need to decide epoch 
    learning_rates = [0.0001,0.001,0.005]  # Learning rates to test
    activation_function = 'sigmoid'  # Fixed activation function

    for lr in learning_rates:
        # Create a new run for each learning rate
        with wandb.init(project="HousingData-3_3-prediction-learning-rate-curves", name=f"lr_{lr}", reinit=True):
            # Data for plotting
            epochs_data = []
            train_losses = []
            val_losses = []

            for epochs in epochs_list:
                # Initialize the MLP model
                mlp = MLPRegressorMultiOutput(input_size=train_x.shape[1],
                          hidden_layers=hidden_layers,
                          output_size=1,
                          learning_rate=lr,
                          activation=activation_function,
                          optimizer='sgd',
                          batch_size=100,
                          epochs=epochs       
                          )  # You can modify this if needed

                # Train the model
                train_y = train_y.reshape(-1, 1)
                mlp.fit(train_x, train_y)

                train_predictions = mlp.predict(train_x)
                train_y = train_y.reshape(-1, 1)
                train_loss =  mlp.compute_loss(test_y,train_predictions)#

                val_predictions = mlp.predict(val_x)
                val_y = val_y.reshape(-1, 1)
                val_loss =  mlp.compute_loss(test_y,val_predictions)#


                # Store data for plotting
                epochs_data.append(epochs)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics to W&B
                wandb.log({
                    "epochs": epochs,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "activation": activation_function
                })

            # Create line plots for accuracy and loss
            wandb.log({
                f"lr_{lr}_loss_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_losses, val_losses],
                    keys=["Train Loss", "Validation Loss"],
                    title=f"Learning Rate: {lr} - Loss vs Epochs",
                    xname="Epochs"
                )
            })

    # Finish W&B run
    wandb.finish()

# https://wandb.ai/kk408-aissms-ioit/HousingData-3_3-prediction-learning-rate-curves
# task3_3_2()



def task3_3_3():
    # Initialize W&B project
    df = pd.read_csv('./data/interim/3/HousingData_standardized.csv')

    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=['MEDV']).values  # Features
    y = df['MEDV'].values  # Target variable
    print("NaN values in the dataset:", np.isnan(df).sum().sum())

    # Define the split sizes
    test_size = int(0.1 * len(df))  # 10% for test
    val_size = int(0.1 * len(df))   # 10% for validation
    # Randomly shuffle the dataset
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.permutation(len(df))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]


    train_x, train_y = X[train_indices], y[train_indices]
    val_x, val_y = X[val_indices], y[val_indices]
    test_x, test_y = X[test_indices], y[test_indices]
    
    # Fixed hyperparameters
    lr = 0.001 #decide 
    hidden_layers =  [10,20,10] #decide  #
    epochs_list = [50,100,150,200,250,300]#,60,70,80,90,100] 
    optimizers = ['sgd', 'mini-batch', 'batch']  # Optimizers to test
    activation_function = 'sigmoid'  # Fixed activation function

    for optimizer in optimizers:
        # Create a new run for each optimizer
        with wandb.init(project="HousingData-3_3-prediction-optimizer-curves", name=f"{optimizer}_optimizer", reinit=True):
            # Data for plotting
            epochs_data = []
            train_losses = []
            val_losses = []

            for epochs in epochs_list:
                # Initialize the MLP model
                mlp = MLPRegressorMultiOutput(input_size=train_x.shape[1],
                          hidden_layers=hidden_layers,
                          output_size=1,
                          learning_rate=lr,
                          activation=activation_function,
                          optimizer=optimizer,
                          batch_size=100,
                          epochs=epochs,
                          )

                # Train the model
                train_y = train_y.reshape(-1, 1)
                mlp.fit(train_x, train_y)

                train_predictions = mlp.predict(train_x)
                train_y = train_y.reshape(-1, 1)
                train_loss =  mlp.compute_loss(test_y,train_predictions)#

                val_predictions = mlp.predict(val_x)
                val_y = val_y.reshape(-1, 1)
                val_loss =  mlp.compute_loss(test_y,val_predictions)#


                # Store data for plotting
                epochs_data.append(epochs)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Log metrics to W&B
                wandb.log({
                    "epochs": epochs,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "activation": activation_function
                })

            # Create line plots for accuracy and loss
            wandb.log({
                    f"{optimizer}_loss_vs_epochs": wandb.plot.line_series(
                    xs=epochs_data,
                    ys=[train_losses, val_losses],
                    keys=["Train Loss", "Validation Loss"],
                    title=f"{optimizer.capitalize()} Optimizer: Loss vs Epochs",
                    xname="Epochs"
                )
            })

    # Finish W&B run
    wandb.finish()
# https://wandb.ai/kk408-aissms-ioit/HousingData-3_3-prediction-optimizer-curves
# task3_3_3()





# 2)Generate a table listing all the hyperparameters tested and their corressponding metrics mentioned above.

def task3_3_2main():
    # Define the sweep configuration for hyperparameter tuning
    sweep_config = {
        'method': 'grid',  # Grid search
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.001, 0.01]
            },
            'epochs': {
                'values': [50,100,150,200, 250]
            },
            'hidden_layers': {
                'values': [[10,20,30,20,10], [10, 20, 10]]
            },
            'activation': {
                'values': ['relu', 'tanh', 'sigmoid', 'linear']
            },
            'optimizer': {
                'values': ['mini-batch', 'sgd', 'batch']
            },
        }
    }
    best_config = {}
    least_mse = 0   
    all_metrics = []  # To store all hyperparameters and corresponding metrics

    # Define the function that will run for each configuration
    def task3_3_2_sweep():
        with wandb.init() as run:
            # Get the current hyperparameters from W&B sweep config
            config = wandb.config
            lr = config.learning_rate
            epoch = config.epochs
            hidden_layer = config.hidden_layers
            activation = config.activation
            optimizer = config.optimizer

            # Load dataset and split it into training, validation, and test sets
            df = pd.read_csv('./data/interim/3/HousingData_standardized.csv')

            # Separate the features (X) and the target variable (y)
            X = df.drop(columns=['MEDV']).values  # Features
            y = df['MEDV'].values  # Target variable
            print("NaN values in the dataset:", np.isnan(df).sum().sum())

            # Define the split sizes
            test_size = int(0.1 * len(df))  # 10% for test
            val_size = int(0.1 * len(df))   # 10% for validation
            # Randomly shuffle the dataset
            np.random.seed(42)  # Set seed for reproducibility
            indices = np.random.permutation(len(df))
            test_indices = indices[:test_size]
            val_indices = indices[test_size:test_size + val_size]
            train_indices = indices[test_size + val_size:]


            train_x, train_y = X[train_indices], y[train_indices]
            val_x, val_y = X[val_indices], y[val_indices]
            test_x, test_y = X[test_indices], y[test_indices]

            # Initialize MLP model using the current sweep configuration
            input_size = train_x.shape[1]
            output_size = 1 # Number of unique classes

            mlp = MLPRegressorMultiOutput(input_size=input_size, 
                      hidden_layers=hidden_layer, 
                      output_size=output_size,#1
                      learning_rate=lr, 
                      activation=activation,
                      optimizer=optimizer,
                      epochs=epoch,                      
                      batch_size=100,
                      )

            # Train the model
            train_y = train_y.reshape(-1, 1)
            mlp.fit(train_x, train_y)

            # Validate the model
            val_predictions = mlp.predict(val_x)
            val_y = val_y.reshape(-1, 1)
            
            val_mse =  mlp.MSE(val_y,val_predictions)     
            val_rmse = mlp.RMSE(val_y,val_predictions)  
            val_rsquared = mlp.Rsquared(val_y, val_predictions)
           
           
            # Log validation metrics to W&B
            wandb.log({
                "learning_rate": lr,
                "epochs": epoch,
                "hidden_layers": hidden_layer,
                "activation": activation,
                "optimizer": optimizer,
                "validation_mse": val_mse,
                "validation_rmse": val_rmse,
                "validation_R-squared": val_rsquared
            })

             # Save hyperparameters and metrics for the table
            all_metrics.append({
                "learning_rate": lr,
                "epochs": epoch,
                "hidden_layers": hidden_layer,
                "activation": activation,
                "optimizer": optimizer,
                "validation_mse": val_mse,
                "validation_rmse": val_rmse,
                "validation_R-squared": val_rsquared
            })

            # Track the best model
            nonlocal least_mse, best_config
            if least_mse < val_mse:
                least_mse = val_mse
                best_config = {
                    "learning_rate": lr,
                    "epochs": epoch,
                    "hidden_layers": hidden_layer,
                    "activation": activation,
                    "optimizer": optimizer,
                    "validation_mse": val_mse,
                    "validation_rmse": val_rmse,
                    "validation_R-squared": val_rsquared
                }




    # Initialize the W&B sweep and run the agent
    sweep_id = wandb.sweep(sweep_config, project="HousingData-prediction")
    wandb.agent(sweep_id, function=task3_3_2_sweep)


    print("#####Best Model Configuration:") ###
    for key, value in best_config.items():
        print(f"{key}: {value}")

    # Generate a table listing all hyperparameters tested and their metrics
    df_metrics = pd.DataFrame(all_metrics)
    print("\nHyperparameter Tuning Results:")
    print(df_metrics)

    # Optionally, save the table to a CSV for reporting
    df_metrics.to_csv('./assignments/3/hyperparamerter_tuning/Housingdata_3_2hyperparameter_tuning_results.csv', index=False) #(2.3.2 - 5 marks)

# GO to link ,sort by mse ,take hyper-para having least mse
# https://wandb.ai/kk408-aissms-ioit/HousingData-prediction
# task3_3_2main()  #kept this case intentionally: , using sgd lr=0.01 gives nan, so reduce lr=0.001



# 3- 3
# Q. Report the parameters for the best model that you get
# After sorting the models based on validation Mean Squared Error (MSE) and selecting the one with the lowest MSE, the best hyperparameters for the model are as follows:
# Learning Rate (lr): 0.001
# Optimizer: Stochastic Gradient Descent (SGD)
# Activation Function: Sigmoid
# Hidden Layer Configuration: [10, 20, 10] (indicating the number of neurons in each hidden layer)
# Epochs: 200


def task3_4():    

    # Load the standardized dataset from a single CSV file
    df = pd.read_csv('./data/interim/3/HousingData_standardized.csv')

    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=['MEDV']).values  # Features
    y = df['MEDV'].values  # Target variable
    print("NaN values in the dataset:", np.isnan(df).sum().sum())

    # Define the split sizes
    test_size = int(0.1 * len(df))  # 10% for test
    val_size = int(0.1 * len(df))   # 10% for validation
    # Randomly shuffle the dataset
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.permutation(len(df))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]


    train_x, train_y = X[train_indices], y[train_indices]
    val_x, val_y = X[val_indices], y[val_indices]
    test_x, test_y = X[test_indices], y[test_indices]


    # Define the model parameters
    input_size = train_x.shape[1]  # Number of features
    output_size = 1  # For regression
    hidden_layers = [10,20,10]  # Example hidden layer sizes


    mlp =MLPRegressorMultiOutput(input_size=input_size,
                hidden_layers=[10,20,10],
                output_size=1,
                learning_rate=0.001,
                activation='sigmoid',
                optimizer='sgd',
                batch_size=100,
                epochs=200
                )

    # Fit the model on the training data
    print("Input contains NaN:", np.isnan(train_x).sum())
    train_y = train_y.reshape(-1, 1)
    mlp.fit(train_x, train_y)

    # Make predictions on the test data
    predictions = mlp.predict(test_x)

    # Compute metrics
    test_y = test_y.reshape(-1, 1)
    mse =  mlp.MSE(test_y,predictions)
    print(f'Mean Squared Error: {mse}')

    mae = np.mean(np.abs(test_y - predictions))  # Calculate MAE
    print(f'Mean Absolute Error: {mae}')

    rmse = mlp.RMSE(test_y, predictions)
    print(f'Root Mean Squared Error: {rmse}')

    rsquared = mlp.Rsquared(test_y, predictions)
    print(f'R-squared: {rsquared}')



# Mean Squared Error: [4.11406969]
# Mean Absolute Error: 1.4727632710457212
# Root Mean Squared Error: [2.02831696]
# R-squared: [0.93529815]
# task3_4()

###################################################################################
# 3.5 Mean Squared Error vs Binary Cross Entropy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_diabetes_data():
    # Load the dataset
    df = pd.read_csv('./data/external/diabetes.csv')

    # Handle missing or invalid values (replace zeros with NaN in specific columns)
    columns_to_replace_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_replace_zeros] = df[columns_to_replace_zeros].replace(0, np.nan)

    # Fill NaN values with the median of the respective columns
    for column in columns_to_replace_zeros:
        df[column].fillna(df[column].median(), inplace=True)

    # Normalize the data (excluding the 'Outcome' column)
    feature_columns = df.columns[:-1]  # All columns except the last one ('Outcome')
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Save the normalized dataset to a new CSV file
    df.to_csv('./data/interim/3/diabetes_norm.csv', index=False)

    print("Data preprocessing completed. Normalized data saved to './data/interim/3/diabetes_norm.csv'.")

# Call the function
preprocess_diabetes_data()



# Load the diabetes dataset
df = pd.read_csv('./data/interim/3/diabetes_norm.csv')
# Features and labels
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
        'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y = df['Outcome'].values.reshape(-1, 1)

# Set the random seed for reproducibility
np.random.seed(42)

# Shuffle the dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Calculate the sizes of the splits
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))

# Split the data
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]
X_test = X[test_indices]
y_test = y[test_indices]

def task3_5_1():

    # Create two models: one with BCE loss and one with MSE loss
    model_bce = MLPRegressorMultiOutputWithLoss(input_size=X_train.shape[1], hidden_layers=[5], output_size=1,
                                                learning_rate=0.01, activation='relu', epochs=100, loss_function='bce')
    # [8, 5, 1]
    model_mse = MLPRegressorMultiOutputWithLoss(input_size=X_train.shape[1], hidden_layers=[5], output_size=1,
                                                learning_rate=0.01, activation='relu', epochs=100, loss_function='mse')
    # [8, 5, 1]
    # Fit both models
    print("Training BCE model...")
    model_bce.fit(X_train, y_train)

    print("Training MSE model...")
    model_mse.fit(X_train, y_train)

    # Evaluate both models
    loss_bce = model_bce.evaluate(X_test, y_test)
    loss_mse = model_mse.evaluate(X_test, y_test)

    print(f"BCE Model Loss: {loss_bce}")
    print(f"MSE Model Loss: {loss_mse}")

# task3_5_1()


def task3_5_2():

    losses_bce = []
    losses_mse = []

    # Fit both models at specified epochs
    for epoch in range(10, 141, 10):
        print(f"Training BCE model - Epoch: {epoch}")
        
        # Create the model with BCE loss for the current epoch
        model_bce = MLPRegressorMultiOutputWithLoss(
            input_size=X_train.shape[1],
            hidden_layers=[5],
            output_size=1,
            learning_rate=0.01,
            activation='relu',
            epochs=epoch,
            loss_function='bce'
        )
        model_bce.fit(X_train, y_train)
        loss_bce = model_bce.evaluate(X_val, y_val)
        losses_bce.append(loss_bce)  # Get the latest loss for BCE

        print(f"Training MSE model - Epoch: {epoch}")
        
        # Create the model with MSE loss for the current epoch
        model_mse = MLPRegressorMultiOutputWithLoss(
            input_size=X_train.shape[1],
            hidden_layers=[5],
            output_size=1,
            learning_rate=0.01,
            activation='relu',
            epochs=epoch,
            loss_function='mse'
        )
        model_mse.fit(X_train, y_train)
        loss_mse = model_mse.evaluate(X_val, y_val)
        losses_mse.append(loss_mse)  # Get the latest loss for MSE

    # Define the epochs for plotting
    epochs_to_plot = np.arange(10, 141, 10)  # 10, 20, ..., 130


    # Plotting the loss vs epochs for MSE
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_to_plot, losses_mse, label='MSE Loss', color='orange', marker='o')
    plt.title('MSE Loss vs Epochs (Every 10 Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs_to_plot)
    plt.legend()
    plt.grid()
    plt.savefig('./assignments/3/figures/task3_5_mse_lossvsepoch.png')
    # plt.show()

    # Plotting the loss vs epochs for BCE
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_to_plot, losses_bce, label='BCE Loss', color='blue', marker='o')
    plt.title('BCE Loss vs Epochs (Every 10 Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs_to_plot)
    plt.legend()
    plt.grid()
    plt.savefig('./assignments/3/figures/task3_5_bce_lossvsepoch.png')
    # plt.show()

    print("losses bce ",losses_bce)
    print("losses bce ",losses_mse)
# task3_5_2()

"""
#               task 3 5 3
Observations :
Convergence:

Both loss functions show convergence, as the loss values generally decrease over time.
BCE loss converges faster initially, showing a sharp drop in the first few epochs.
MSE loss has a more gradual and consistent decrease throughout the epochs.


Stability:

BCE loss shows more fluctuations and oscillations in its values across epochs.
MSE loss appears more stable, with smaller variations between consecutive epochs.


Final loss values:

BCE loss converges to values around 0.5, which is higher than MSE loss.
MSE loss converges to lower values, around 0.16-0.17.


Learning behavior:

BCE loss seems to reach its approximate final value quicker but then oscillates around it.
MSE loss shows a more steady learning curve, gradually approaching its final value.


Sensitivity:

BCE loss appears more sensitive to small changes in the model's predictions, leading to larger fluctuations.
MSE loss seems less sensitive, resulting in a smoother convergence plot.


Interpretation:

The higher final value of BCE loss doesn't necessarily indicate worse performance, as BCE and MSE have different scales and interpretations.
BCE loss is more appropriate for binary classification tasks, while MSE is typically used for regression problems.


Optimization landscape:

The oscillations in BCE loss might suggest a more complex optimization landscape, possibly with local minima or saddle points.
The smoother MSE loss curve could indicate a simpler optimization landscape for this particular problem.


Learning rate effects:
The behavior of both losses might suggest that the learning rate could be further optimized, especially for the BCE model to reduce oscillations.

Difference :
key differences:

Convergence speed: BCE converges faster initially, while MSE shows a more gradual decrease.
Stability: BCE exhibits more fluctuations, while MSE is more stable.
Final loss values: BCE converges to higher values (around 0.5) compared to MSE (around 0.16-0.17).
Learning behavior: BCE reaches its approximate final value quicker but oscillates, while MSE shows a steadier approach.
Sensitivity: BCE appears more sensitive to small changes, resulting in larger fluctuations.
"""


def task3_6():
    # Create the model (assuming MLPRegressorMultiOutputWithLoss is already defined)
    model = MLPRegressorMultiOutputWithLoss(input_size=X_train.shape[1], hidden_layers=[5], output_size=1,
                                            learning_rate=0.01, activation='relu', epochs=130, loss_function='mse')

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate MSE for each data point
    individual_mse = (y_test.flatten() - y_pred.flatten()) ** 2  # Squared error for each point



    # Create a DataFrame for analysis
    results_df = pd.DataFrame({
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten(),
        'MSE': individual_mse
    })

    # Show the results
    print("Results DataFrame:")
    print(results_df)

    # Analyze high MSE and low MSE points
    mean_mse = results_df['MSE'].mean()
    high_mse = results_df[results_df['MSE'] > mean_mse]
    low_mse = results_df[results_df['MSE'] <= mean_mse]

    print("\nData points with high MSE Loss:")
    print(high_mse)

    print("\nData points with low MSE Loss:")
    print(low_mse)

    # Optional: Analyze features associated with high/low MSE points
    high_mse_indices = high_mse.index
    low_mse_indices = low_mse.index

    print("\nAverage Features of High MSE Points:")
    print(X_test[high_mse_indices].mean(axis=0))

    print("\nAverage Features of Low MSE Points:")
    print(X_test[low_mse_indices].mean(axis=0))

    import matplotlib.pyplot as plt

    # Plot MSE distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['MSE'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Mean Squared Error')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("./assignments/3/figures/task3_6_MSEloss.png")
    # plt.show()

"""
OutPut : task 3.6
Results DataFrame:
    Actual  Predicted       MSE
0        1   0.268207  0.535521
1        0   0.469093  0.220048
2        0   0.197268  0.038915
3        0   0.155528  0.024189
4        0   0.204805  0.041945
..     ...        ...       ...
73       0   0.385705  0.148768
74       0   0.011541  0.000133
75       1   0.778927  0.048873
76       1   0.529877  0.221016
77       0   0.046226  0.002137

[78 rows x 3 columns]

Data points with high MSE Loss:
    Actual  Predicted       MSE
0        1   0.268207  0.535521
1        0   0.469093  0.220048
5        1   0.192752  0.651649
14       0   0.394698  0.155786
18       1   0.226730  0.597946
22       0   0.595064  0.354101
27       1   0.417938  0.338796
29       0   0.453807  0.205941
31       1   0.485536  0.264674
33       1   0.510866  0.239252
35       1   0.534071  0.217090
39       1   0.426333  0.329094
40       0   0.586083  0.343493
43       1   0.095607  0.817927
44       0   0.806084  0.649771
46       1   0.288910  0.505649
51       0   0.470447  0.221321
52       1   0.101775  0.806808
53       0   0.435231  0.189426
58       1   0.351213  0.420925
62       1   0.461252  0.290249
66       0   0.683468  0.467128
67       1   0.306596  0.480809
71       0   0.401915  0.161536
73       0   0.385705  0.148768
76       1   0.529877  0.221016

Data points with low MSE Loss:
    Actual  Predicted       MSE
2        0   0.197268  0.038915
3        0   0.155528  0.024189
4        0   0.204805  0.041945
6        0   0.235023  0.055236
7        0   0.072177  0.005210
8        0   0.244455  0.059758
9        0   0.018787  0.000353
10       0   0.115204  0.013272
11       1   0.802422  0.039037
12       0   0.040015  0.001601
13       0   0.084705  0.007175
15       0   0.101791  0.010361
16       1   0.830609  0.028693
17       0   0.183499  0.033672
19       0   0.314265  0.098762
20       1   0.838574  0.026058
21       0   0.062936  0.003961
23       0   0.082268  0.006768
24       1   0.776679  0.049872
25       0   0.152855  0.023365
26       0   0.058612  0.003435
28       0   0.018168  0.000330
30       1   0.838574  0.026058
32       1   0.689724  0.096271
34       0   0.107467  0.011549
36       0   0.132526  0.017563
37       0   0.070699  0.004998
38       0   0.113498  0.012882
41       0   0.168048  0.028240
42       1   0.784930  0.046255
45       0   0.084880  0.007205
47       0   0.016375  0.000268
48       0   0.240010  0.057605
49       0   0.212405  0.045116
50       0   0.274046  0.075101
54       0   0.044816  0.002008
55       0   0.187018  0.034976
56       0   0.052954  0.002804
57       0   0.311795  0.097216
59       1   0.785872  0.045851
60       1   0.768399  0.053639
61       1   0.838574  0.026058
63       0   0.060629  0.003676
64       0   0.144334  0.020832
65       1   0.838574  0.026058
68       0   0.009679  0.000094
69       0   0.199108  0.039644
70       1   0.754737  0.060154
72       0   0.287539  0.082679
74       0   0.011541  0.000133
75       1   0.778927  0.048873
77       0   0.046226  0.002137

Average Features of High MSE Points:
[0.28054299 0.53846154 0.52747253 0.26379599 0.15412352 0.31186094
 0.21398542 0.21602564]

Average Features of Low MSE Points:
[0.20135747 0.49503722 0.48802983 0.2263796  0.14168824 0.29015259
 0.1594955  0.19487179]
"""

"""
Report

Patterns in MSE Loss for test dataset points:

Analyzing the provided data, we can observe several patterns:
a. Distribution of errors:

High MSE points tend to have predictions far from their actual values (0 or 1).
Low MSE points have predictions closer to their actual values.

b. Misclassifications:

Many high MSE points are misclassifications (e.g., predicting ~0.2 for an actual 1, or ~0.8 for an actual 0).
Low MSE points are generally correctly classified or have predictions very close to the decision boundary.

c. Feature differences:
Comparing the average features of high and low MSE points:

High MSE points have slightly higher values for most features.
The first feature shows the largest difference (0.2805 vs 0.2013).
The third and sixth features also show notable differences.

d. Decision boundary:

Points near the decision boundary (predictions around 0.5) tend to have higher MSE, which is expected as these are the most uncertain predictions.

e. Extreme predictions:

Very low MSE is often associated with highly confident correct predictions (e.g., predicting 0.011 for an actual 0).

f. Class imbalance:

There seems to be a slight imbalance, with more 0s than 1s in the dataset.

Reasons for these patterns:

Model confidence: High MSE often results from the model being confidently wrong, while low MSE comes from being either correct or uncertain.
Feature importance: The differences in average feature values suggest that certain features might be more influential in causing misclassifications.
Nonlinear decision boundary: The presence of misclassifications with high confidence suggests the true decision boundary might be nonlinear or complex, which the logistic regression model struggles to capture perfectly.
Data characteristics: The slight class imbalance and feature distributions might contribute to the model's performance variations across different data points.
Limitation of MSE for classification: Using MSE for a binary classification problem can lead to these patterns, as it doesn't directly optimize for the classification boundary like BCE does.

These observations highlight the importance of feature engineering, potentially using a more complex model, and considering alternative metrics for binary classification tasks.
"""


# task3_6()


#############################################################
# 3.7 [BONUS: 15 marks]

def bonusclassi():
    # 1) with classification example
    df = pd.read_csv('./data/interim/3/WineQT_standar.csv')

    # Separate features and labels
    X = df.drop('quality', axis=1).values  # Features (11 attributes)
    y = df['quality'].values  # Labels (Wine Quality)

    # Check unique labels in y
    print("Unique labels in y:", np.unique(y))

    # Convert labels to one-hot encoding (assuming labels range from 3 to 8)
    num_classes = len(np.unique(y))  # Number of unique classes
    y_one_hot = np.eye(num_classes)[y - 3]  # One-hot encoding, assuming labels are [3, 4, 5, 6, 7, 8]
    # (1143, 6)

    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.permutation(len(X))

    # Define sizes for the splits
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    test_size = len(X) - train_size - val_size

    # Split the indices into training, validation, and test
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create the train, validation, and test sets
    train_x, train_y = X[train_indices], y_one_hot[train_indices]
    val_x, val_y = X[val_indices], y_one_hot[val_indices]
    test_x, test_y = X[test_indices], y_one_hot[test_indices]
    input_size = train_x.shape[1]  # Number of features (11) 
    output_size = num_classes  # Number of classes (6 in this case: 3, 4, 5, 6, 7, 8)

    mlp = CommonMLP_bonus(input_size=input_size, hidden_layers=[32,32], output_size=6,  # Example 5-class classification
                output_activation='softmax', loss='cross_entropy', task_type='classification',
                learning_rate=0.01, epochs=100)
    print(train_x.shape, "          -   ",train_y.shape)
    mlp.fit(train_x, train_y)  # y_train_one_hot is one-hot encoded labels
    predictions = mlp.predict(val_x)  # Get the predicted probabilities
    predicted_indices = np.argmax(predictions, axis=1)  # For single-label classification
    actual_indices   = np.argmax(val_y, axis=1)
    accuracy = np.mean(predicted_indices == actual_indices)  # Compare with original labels
    print(f"Classification Validation Accuracy: {accuracy * 100:.2f}%")




def bonusregress():
    df = pd.read_csv('./data/interim/3/HousingData_standardized.csv')
    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=['MEDV']).values  # Features
    y = df['MEDV'].values  # Target variable
    print("NaN values in the dataset:", np.isnan(df).sum().sum())

    # Define the split sizes
    test_size = int(0.1 * len(df))  # 10% for test
    val_size = int(0.1 * len(df))   # 10% for validation
    # Randomly shuffle the dataset
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.permutation(len(df))
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]


    train_x, train_y = X[train_indices], y[train_indices]
    val_x, val_y = X[val_indices], y[val_indices]
    test_x, test_y = X[test_indices], y[test_indices]


    # Define the model parameters
    input_size = train_x.shape[1]  # Number of features
    output_size = 1  # For regression
    hidden_layers = [10,20,30,20,10] 


    # For regression:
    mlp = CommonMLP_bonus(input_size=train_x.shape[1], hidden_layers=[10,20,10], output_size=1,  # Regression output size
            output_activation='linear', loss='mse', task_type='regression',
            learning_rate=0.01, epochs=100)

    train_y = train_y.reshape(-1, 1)
    print(train_x.shape,"    ",train_y.shape)
    mlp.fit(train_x, train_y)


    # Make predictions on the test data
    predictions = mlp.predict(test_x)
    # Compute metrics
    test_y = test_y.reshape(-1, 1)
    mse =  mlp.compute_loss(test_y,predictions)
    print(f'Regression Mean Squared Error: {mse}')

# bonusregress()
def task3_7():
    bonusclassi()
    bonusregress()

task3_7() #made compatable with both classification and regression ,gives same MSE as per single class





###################################################


# 4 AutoEncoders



from models.AutoEncoders_usingmlp3.AutoEncoders import *
def task4_2():

    # Load the datasets
    df_train = pd.read_csv('./data/interim/1/spotify_train.csv')
    df_test = pd.read_csv('./data/interim/1/spotify_test.csv')
    df_val = pd.read_csv('./data/interim/1/spotify_val.csv')

    # Separate features (X) and target (y)
    X_train = df_train.iloc[:, :-1].values  # All columns except the last one (features)
    y_train = df_train.iloc[:, -1].values   # Last column (genre)
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    X_val = df_val.iloc[:, :-1].values
    y_val = df_val.iloc[:, -1].values


    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    mlp =AutoEncoder(
        input_size=X_train.shape[1],
        before=[10,8],
        latent_size=5,
        after=[8,10],
        output_size=X_train.shape[1],
        learning_rate=0.01,
        activation='relu',
        optimizer='mini-batch',
        batch_size=10,
        epochs=100
    )

    mlp.fit(X_train,X_train) #4.2 trained encoder 

    # Obtain the latent vectors from the validation set
    latent_reduced_val, pred_val = mlp.get_latent(X_val)  # After forward propagation for validation
    mse_val = np.mean(np.abs(X_val - pred_val))  # Loss computation for validation
    print("Loss of validation data:", mse_val)

    # print("Latent representations of validation data:", latent_reduced_val)

    # Create a DataFrame to save the latent vectors of validation set
    encoded_val_df = pd.DataFrame(latent_reduced_val, columns=[f'latent_{i+1}' for i in range(latent_reduced_val.shape[1])])
    encoded_val_df['genre'] = y_val  # Add the genre column for validation set

    # Save the DataFrame to a CSV file for validation set
    encoded_val_df.to_csv('./data/interim/3/encoded_val.csv', index=False)
    print("Encoded validation data saved to './data/interim/3/encoded_val.csv'.")

    # Obtain the latent vectors from the training set
    latent_reduced_train, pred_train = mlp.get_latent(X_train)  # After forward propagation for training
    mse_train = np.mean(np.abs(X_train - pred_train))  # Loss computation for training
    print("Loss of training data:", mse_train)

    print("Latent representations of training data:", latent_reduced_train)

    # Create a DataFrame to save the latent vectors of training set
    encoded_train_df = pd.DataFrame(latent_reduced_train, columns=[f'latent_{i+1}' for i in range(latent_reduced_train.shape[1])])
    encoded_train_df['genre'] = y_train  # Add the genre column for training set

    # Save the DataFrame to a CSV file for training set
    encoded_train_df.to_csv('./data/interim/3/encoded_train.csv', index=False)
    print("Encoded training data saved to './data/interim/3/encoded_train.csv'.")

# Encoded validation data saved to './data/interim/3/encoded_val.csv'.
# task4_2()

from models.knn.knn import *

def task4_3():
    # Load the encoded datasets
    train_df = pd.read_csv('./data/interim/3/encoded_train.csv')
    val_df = pd.read_csv('./data/interim/3/encoded_val.csv')

    # Separate the target genres from the training and validation data
    Y_train = train_df['genre'].values
    Y_val = val_df['genre'].values

    # Select only the latent features (PCs)
    X_train = train_df[[f'latent_{i+1}' for i in range(5)]].values  # Adjust the number of PCs as needed
    X_val = val_df[[f'latent_{i+1}' for i in range(5)]].values  # Adjust the number of PCs as needed


    # Initialize and train the KNN model
    knn = KNearestNeighbours(k=15, distance_metric='manhattan', prediction_type="weighted_sum")
    knn.fit(X_train, Y_train)

    # Validate the model on the validation set
    metrics = knn.validate(X_val, Y_val)
    print("Metrics:", metrics)


""" 
# assignment 3 , latent vector of 5 dimension
# accuracy :  0.11 
# Metrics: {'macro_precision': 0.10756549391996577, 'macro_recall': 0.11015949839175487, 'macro_f1': 0.10757789952297923, 'micro_precision': 0.11, 'micro_recall': 0.11, 'micro_f1': 0.11, 'accuracy': 0.9843859649122807}



# recuded pca dataset : assignment 2 ,top 5  principal component
# accuracy :  0.104
# metrices  {'macro_precision': 0.09777025948927576, 'macro_recall': 0.10554057758656253, 'macro_f1': 0.09906639475522733, 'micro_precision': 0.104, 'micro_recall': 0.104, 'micro_f1': 0.104, 'accuracy': 0.984280701754386}


# full dataset:  assignment 1 ,all 13 features
# accuracy :  0.1794736842105263
# metrices  {'macro_precision': 0.18147587154189257, 'macro_recall': 0.18053097639379498, 'macro_f1': 0.17786395033004604, 'micro_precision': 0.1794736842105263, 'micro_recall': 0.1794736842105263, 'micro_f1': 0.1794736842105263, 'accuracy': 0.9856048014773776}

# Comparing the metrics across the three assignments reveals interesting insights about the performance of different dimensionality reduction techniques and their impact on the KNN model:

# Accuracy:

# Assignment 1 (full dataset): 98.56%
# Assignment 2 (PCA): 98.43%
# Assignment 3 (Autoencoder): 98.44%

# The accuracy remains high and very similar across all three approaches, with the full dataset slightly outperforming the reduced versions.
# F1 Score (Macro):

# Assignment 1: 0.1779
# Assignment 2: 0.0991
# Assignment 3: 0.1076

# The full dataset significantly outperforms both dimensionality reduction techniques in terms of F1 score.
# Precision and Recall:
# Both macro and micro precision/recall follow a similar pattern to the F1 score, with the full dataset performing best, followed by the autoencoder, and then PCA.

# Analysis:

# Information Preservation: The full dataset retains all original information, leading to better overall performance. Both dimensionality reduction techniques (PCA and autoencoder) lose some information, resulting in slightly lower performance.
# Autoencoder vs. PCA: The autoencoder slightly outperforms PCA, suggesting it captures more relevant information in its 5-dimensional latent space compared to PCA's top 5 principal components.
# Trade-off: While dimensionality reduction slightly decreases performance, it offers benefits like reduced computational complexity and potentially better generalization on unseen data.
# Class Imbalance: The high accuracy but lower F1 scores across all methods suggest a class imbalance in the dataset, which affects the model's performance on minority classes.
# Feature Importance: The performance drop in reduced datasets indicates that some less prominent but still important features for classification are lost during dimensionality reduction.
"""
# task4_3()


def task4_4():
    df_train = pd.read_csv('./data/interim/1/spotify_train.csv')
    df_test = pd.read_csv('./data/interim/1/spotify_test.csv')
    df_val = pd.read_csv('./data/interim/1/spotify_val.csv')

    # Separate features (X) and target (y)
    X_train = df_train.iloc[:, :-1].values  # All columns except the last one (features)
    y_train = df_train.iloc[:, -1].values   # Last column (genre)
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    X_val = df_val.iloc[:, :-1].values
    y_val = df_val.iloc[:, -1].values


    # Check unique labels in y_train (for one-hot encoding)
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))  # Ensure all splits use same labels
    # print("Unique labels in the dataset:", unique_labels)


    # Convert labels to one-hot encoding
    num_classes = len(unique_labels)  # Number of unique classes
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Map labels to consecutive integers
    y_train_one_hot = np.eye(num_classes)[np.vectorize(label_map.get)(y_train)]
    y_val_one_hot = np.eye(num_classes)[np.vectorize(label_map.get)(y_val)]
    y_test_one_hot = np.eye(num_classes)[np.vectorize(label_map.get)(y_test)]

    # Verify the shape of one-hot encoded labels
    # print("y_train_one_hot shape:", y_train_one_hot.shape)
    # print("y_val_one_hot shape:", y_val_one_hot.shape)
    # print("y_test_one_hot shape:", y_test_one_hot.shape)

    y_train =y_train_one_hot
    y_val =y_val_one_hot
    y_test =y_test_one_hot


    input_size = X_train.shape[1]  # Number of features (11) 
    output_size = num_classes

    hidden_layers = [32, 64 ,64,32 ]

    mlp = MLP(input_size=input_size, 
                hidden_layers=hidden_layers, 
                output_size=output_size,
                output_activation='softmax',  # Use softmax for multi-class classification
                loss='cross_entropy',  # Use cross-entropy loss
                learning_rate=0.01, 
                epochs=100,
                activation='relu',
                batch_size=10,
                optimizer="mini-batch" 
                )

    print(X_train.shape, "   -- - - ",y_train.shape)

    mlp.fit(X_train[:10000], y_train[:10000]) 
    mlp.fit(X_train[10000:20000], y_train[10000:20000])
    mlp.fit(X_train[20000:30000], y_train[20000:30000]) 
    mlp.fit(X_train[30000:40000], y_train[30000:40000]) 
    mlp.fit(X_train[40000:50000], y_train[40000:50000]) 
    mlp.fit(X_train[50000:60000], y_train[50000:60000]) 
    mlp.fit(X_train[60000:70000], y_train[60000:70000]) 
    predictions = mlp.predict(X_val)  
    predicted_indices = np.argmax(predictions, axis=1)  # For single-label classification
    actual_indices   = np.argmax(y_val, axis=1)
    accuracy = np.mean(predicted_indices == actual_indices)  # Compare with original labels
    print(f"Accuracy: {accuracy * 100:.2f}%")

    metrices =performance_metrices(predicted_indices,actual_indices)
    print("metrices ",metrices)


# Accuracy: 14.27%
# metrices  {'macro_precision': 0.14353673655912333, 'macro_recall': 0.13059697170457843, 'macro_f1': 0.12868623986647845, 'micro_precision': 0.14271929824561402, 'micro_recall': 0.14271929824561402, 'micro_f1': 0.14271929824561402, 'accuracy': 0.9849599876885196}
# Accuracy:

# MLP: 98.50%
# KNN (full dataset): 98.56%
# KNN (autoencoder): 98.44%
# KNN (PCA): 98.43%

# The MLP classifier achieves high accuracy, comparable to the KNN models.
# Micro F1 Score:

# MLP: 0.1427
# KNN (full dataset): 0.1795
# KNN (autoencoder): 0.1100
# KNN (PCA): 0.1040

# The MLP outperforms the dimensionality-reduced KNN models but falls short of the full-dataset KNN.
# Macro F1 Score:

# MLP: 0.1287
# KNN (full dataset): 0.1779
# KNN (autoencoder): 0.1076
# KNN (PCA): 0.0991

# Similar trend as micro F1, with MLP performing better than reduced-dimension KNN but worse than full-dataset KNN.
# Precision and Recall:
# The MLP shows balanced precision and recall, performing better than the reduced-dimension KNN models but not reaching the level of the full-dataset KNN.

# Insights:

# Model Complexity: The MLP, being a more complex model, captures nonlinear relationships better than KNN on reduced datasets, leading to improved performance.
# Feature Utilization: MLP's performance suggests it effectively utilizes the full feature set, unlike KNN which performs best with all features but degrades with dimensionality reduction.
# Generalization: MLP's slightly lower accuracy but higher F1 scores compared to full-dataset KNN might indicate better generalization and handling of class imbalance.
# Dimensionality Reduction Trade-off: The MLP results reinforce that dimensionality reduction techniques, while useful for KNN, may not always be necessary for more sophisticated models like neural networks.
# Class Imbalance Handling: The MLP seems to handle class imbalance slightly better than KNN, as evidenced by the improved F1 scores despite similar accuracy.
# Model Choice: For this dataset, the full-feature KNN slightly edges out MLP in overall performance, but MLP shows promise in balancing precision and recall.

# task4_4()


