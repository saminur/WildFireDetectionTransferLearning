# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:28:42 2024

@author: samin
"""

import timm
import torch.nn as nn
import torch


# Load the pretrained Swin Transformer model
# model = timm.create_model('swin_base_patch4_window12_384', pretrained=True)
# model = timm.create_model('swin_tiny_patch4_window7_256', pretrained=True)
flag_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
if not flag_cuda:
    print('Using CPU')
else:
    print('Using GPU')
    
path_model = 'swin_tiny_patch4_window7_224_22k.pth'
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

state_dict = torch.load(path_model, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
# Move the model to the GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# Modify the final layers for binary classification
num_features = model.head.in_features
# model.head = nn.Linear(num_features, 2)  # Assuming 2 classes (fire, no fire)
model.head = nn.Sequential(
    nn.Linear(num_features, 32),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(32, 2)
   )
# Make sure to use the model in evaluation mode
model.eval()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split

# Define transformations
# transform = transforms.Compose([
#     transforms.CenterCrop((254, 254)),  # Crop the center of the image to 254x254
#     transforms.Pad((65, 65, 65, 65)),  # Pad the image to 384x384
#     transforms.ToTensor(),
# ])
# Define transformations
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 254x254
    transforms.ToTensor(),
])


# Load the dataset
dataset = datasets.ImageFolder('FLAME_Dataset/Training', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

num_epochs = 10


for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs_flattened = outputs.view(outputs.size(0), -1)  # Reshape to [batch_size, 7*7*2]
        print(outputs_flattened.shape)
        loss = criterion(outputs_flattened, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            outputs_flattened = outputs.view(outputs.size(0), -1)  # Reshape to [batch_size, 7*7*2]
            _, predicted = torch.max(outputs_flattened.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_accuracy:.4f}')
    
#save the trained model
path_model = 'swin_tiny_flame_pretrained.pth'
torch.save(model.state_dict(),path_model)
    
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import torch

# Assuming you have already trained your model and have the trained model
# and test dataset available

# Define transformations for the test dataset
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
])


# Load the test dataset
test_dataset = datasets.ImageFolder('FLAME_Dataset/Test', transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Initialize lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []
import torch.nn.functional as F
# Iterate over the test dataset and collect true labels and predicted probabilities
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        outputs_flattened = outputs.view(outputs.size(0), -1)
        probabilities = F.softmax(outputs_flattened, dim=1)

        # Convert the probabilities tensor to a numpy array for further processing
        predicted = probabilities[:, 1].cpu().detach().numpy()
        # _, predicted = torch.max(outputs_flattened.data, 1)
        predicted_probs.extend(predicted) 
        true_labels.extend(labels.numpy())


# Calculate precision-recall curve and AUC
precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
pr_auc = auc(recall, precision)
average_precision = average_precision_score(true_labels, predicted_probs)

# Plot the precision-recall curve
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
plt.show()

from sklearn import metrics
# Plot the AUC curve
fpr, tpr, _ = metrics.roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Calculate accuracy
y_pred = [1 if p > 0.5 else 0 for p in predicted_probs]  # Convert probabilities to binary predictions
accuracy = accuracy_score(true_labels, y_pred)

# Calculate F1 score
f1 = f1_score(true_labels, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Assuming y_true and y_pred are the true labels and predicted labels
y_true = np.array(true_labels)
y_pred = np.array(y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
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


