# -*- coding: utf-8 -*-
"""committee_swin_wildfire_latest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z7k9nQtjqKASOdg0Oc2Yxu1i9ixOCIiu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:06:13 2024

@author: samin
"""

from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_model():
    model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    num_features = model.config.hidden_size
    for param in model.parameters():
        param.requires_grad = False
    # Replace the classifier head
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    model.to(device)
    return model

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3*256*256),  # Assuming images are 256x256 with 3 channels
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Initialize the GAN generator
gan_generator = Generator()

# Define GAN-based augmentation function
def gan_augmentation(image):
    # Convert PIL Image to PyTorch Tensor
    image_tensor = transforms.ToTensor()(image)
    
    # Generate a random noise vector
    # noise = truncated_noise_sample(truncation=0.7, batch_size=1, seed=None)
    # noise = torch.tensor(noise, dtype=torch.float32)
    # # noise_gen = gan_generator(noise)
    noise = torch.randn(1, 50)
    with torch.no_grad():
        fake_image_tensor = transforms.ToPILImage()(gan_generator(noise).view(3, 256, 256))
    
    # Convert fake_image_tensor to PyTorch Tensor
    fake_image_tensor = transforms.ToTensor()(fake_image_tensor)
    
    # Combine the fake image with the original image
    augmented_image_tensor = torch.add(image_tensor, fake_image_tensor)
    
    # Convert back to PIL Image
    augmented_image = transforms.ToPILImage()(augmented_image_tensor)
    
    return augmented_image

# Initialize three models
models = [initialize_model() for _ in range(3)]

from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
import random


# Define transformations including GAN-based augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda x: gan_augmentation(x)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = datasets.ImageFolder('FLAME_Dataset/Training', transform=transform)

# Create three random subsets with replacement
full_size = len(dataset)
subset_indices = [random.choices(range(full_size), k=full_size) for _ in range(3)]
subsets = [Subset(dataset, indices) for indices in subset_indices]

# Create a DataLoader for each subset
batch_size = 32
loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]

num_epochs = 20
patience = 10  # Number of epochs with no improvement after which training will be stopped
min_delta = 0.001  # Minimum change in validation loss to be considered as improvement

for model_index, model in enumerate(models):
    subset = subsets[model_index]
    full_size = len(subset)

    # Split the subset into training and validation sets
    train_size = int(0.8 * full_size)
    val_size = full_size - train_size
    train_subset, val_subset = random_split(subset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    train_losses = []
    val_losses = []

    from sklearn.metrics import accuracy_score

    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs.logits, labels)
            predicted_label = outputs.logits.argmax(-1)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            all_train_preds.extend(predicted_label.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_losses.append(train_loss / len(train_loader))
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        print(f'Model {model_index+1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for images_val, labels_val in val_loader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                logits_validation = model(images_val).logits
                loss_val = criterion(logits_validation, labels_val)
                val_loss += loss_val.item()
                predicted_label = logits_validation.argmax(-1)
                all_val_preds.extend(predicted_label.cpu().numpy())
                all_val_labels.extend(labels_val.cpu().numpy())

        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_losses.append(val_loss / len(val_loader))
        print(f'Model {model_index+1}, Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f'Model {model_index+1}: Validation loss did not improve for {patience} epochs. Stopping training.')
                break

import matplotlib.pyplot as plt
# Plotting the training and validation losses
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to apply majority voting
def majority_vote(predictions):
    """
    Aggregate predictions using majority voting.
    predictions: a list of tensors with predictions from each model.
    """
    stacked_predictions = torch.stack(predictions, dim=0)
    majority_votes, _ = torch.mode(stacked_predictions, dim=0)
    return majority_votes


# Load the dataset
test_dataset = datasets.ImageFolder('FLAME_Dataset/Test', transform=transform_test)

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Initialize lists to store true labels and predicted probabilities
all_labels = []
all_preds = []
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, auc
import numpy as np

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        predictions = [model(images).logits.argmax(dim=1) for model in models]
        final_preds = majority_vote(predictions)
        all_preds.extend(final_preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# # Iterate over the test dataset and collect true labels and predicted probabilities
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         logits_test = model(inputs).logits
#         predicted_label = logits_test.argmax(-1)
#         all_preds.extend(predicted_label.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)

# Calculate F1 score
f1 = f1_score(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# Calculate AUC
auc_score = roc_auc_score(all_labels, all_preds)
print(f'AUC: {auc_score:.4f}')

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(f'Confusion Matrix:\n{cm}')
classes =['Fire','No Fire']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


# Calculate ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
