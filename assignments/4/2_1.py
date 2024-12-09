import os
from PIL import Image

def load_mnist_data(base_path):
    data = {'train': [], 'val': [], 'test': []}
    labels = {'train': [], 'val': [], 'test': []}
    
    # Loop over each data split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        
        # Loop over each label folder within the split directory
        for label_folder in os.listdir(split_path):
            label_path = os.path.join(split_path, label_folder)
            
            # Check if the path is a directory
            if os.path.isdir(label_path):
                # Determine the label by counting the digits in the folder name
                label_count = len(label_folder) if label_folder.isdigit() else 0
                
                # Load each image in the folder
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    
                    # Open image, load data, then close immediately
                    with Image.open(img_path) as img:
                        data[split].append(img.copy())  # Use img.copy() to load data into memory
                    labels[split].append(label_count)
                    
    return data, labels



base_path = 'C:/Users/Pavan/Desktop/smai/Assignment/smai-m24-assignments-codepk37/data/external/double_mnist'
data, labels = load_mnist_data(base_path) #PIL format, label length

# Example to access train data
train_images = data['train']
train_labels = labels['train']

val_images = data['val']
val_labels = labels['val']

test_images = data['test']
test_labels = labels['test']
