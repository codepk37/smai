
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##
import numpy as np

def compute_accuracy(y_true, y_pred):
    """Compute accuracy as the ratio of correct predictions to total predictions."""
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy






#https://chatgpt.com/share/941a6a82-2ab8-4a7c-a7c3-a20efb2e2705
def compute_confusion_matrix(y_true, y_pred):
    # Get unique labels
    labels = np.unique(np.concatenate((y_true, y_pred)))
    
    # Create a mapping from label to index
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    # Initialize confusion matrix
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    
    # Fill in the confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_label]
        cm[true_index, pred_index] += 1
    
    return cm, labels

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.savefig("matrix.png")
    plt.show()
    plt.close()

def compute_label_matrices(cm, labels):
    label_matrices = {}
    
    for i, label in enumerate(labels):
        # Initialize TP, TN, FP, FN
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        # Store 2x2 matrix for this label
        label_matrices[label] = {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }
    
    return label_matrices


def calculate_macro_micro_metrics_with_accuracy(label_matrices):
    # Initialize sums for micro metrics and accuracy calculation
    micro_TP = 0
    micro_FP = 0
    micro_FN = 0
    micro_TN = 0
    
    # Lists to store individual label metrics for macro calculation
    precisions = []
    recalls = []
    f1_scores = []
    
    # Iterate over all labels to calculate macro and micro components
    for label, metrics in label_matrices.items():
        TP = metrics["TP"]
        TN = metrics["TN"]
        FP = metrics["FP"]
        FN = metrics["FN"]
        
        # Calculate precision, recall for this label
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Append to macro lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        
        # Update micro sums
        micro_TP += TP
        micro_TN += TN
        micro_FP += FP
        micro_FN += FN
    
    # Calculate macro metrics
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    
    # Calculate micro metrics
    micro_precision = micro_TP / (micro_TP + micro_FP) if (micro_TP + micro_FP) > 0 else 0
    micro_recall = micro_TP / (micro_TP + micro_FN) if (micro_TP + micro_FN) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Calculate accuracy
    accuracy = (micro_TP + micro_TN) / (micro_TP + micro_TN + micro_FP + micro_FN) # (TP + TN) / (TP + TN + FP + FN)
    
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "accuracy": accuracy
    }




# Example usage

y_true = np.array(['grunge', 'grunge', 'club', 'hardstyle', 'indian', 'club', 'hardstyle', 'indian'])
y_pred = np.array(['breakbeat', 'grunge', 'breakbeat', 'hardstyle', 'indian', 'breakbeat', 'hardstyle', 'indian'])
#NOTE: works for all predictions since class label is used

def performance_metrices(y_true,y_pred):
    cm, labels = compute_confusion_matrix(y_true, y_pred)
    # Print confusion matrix
    # print(f'Confusion Matrix:\n{cm}')

    # Plot confusion matrix
    # plot_confusion_matrix(cm, labels) #run only this file
    
    # Compute label-specific matrices
    label_matrices = compute_label_matrices(cm, labels)
    # Print matrices for each label
    # for label, metrics in label_matrices.items():
    #     print(f'\nMetrics for label "{label}":')
    #     print(f'TP: {metrics["TP"]}')
    #     print(f'TN: {metrics["TN"]}')
    #     print(f'FP: {metrics["FP"]}')
    #     print(f'FN: {metrics["FN"]}')

    metrics = calculate_macro_micro_metrics_with_accuracy(label_matrices)
    # print("Macro Precision: ", metrics["macro_precision"])
    # print("Macro Recall: ", metrics["macro_recall"])
    # print("Macro F1: ", metrics["macro_f1"])
    # print("Micro Precision: ", metrics["micro_precision"])
    # print("Micro Recall: ", metrics["micro_recall"])
    # print("Micro F1: ", metrics["micro_f1"])
    # print("Accuracy: ", metrics["accuracy"])
    return metrics



if __name__=='__main__':
    performance_metrices(y_true,y_pred)