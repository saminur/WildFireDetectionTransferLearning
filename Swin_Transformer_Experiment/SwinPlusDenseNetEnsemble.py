# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:59:24 2024

@author: samin
"""

from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import densenet201

from sklearn.metrics import accuracy_score




transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Load the dataset
dataset = datasets.ImageFolder('FLAME_Dataset/Training', transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the Swin Transformer model and image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model_swin = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_swin.to(device)
# Load the DenseNet201 model
model_densenet = densenet201(pretrained=True)
model_densenet.to(device)

# Freeze all parameters in both models
for param in model_swin.parameters():
    param.requires_grad = False
for param in model_densenet.parameters():
    param.requires_grad = False

# Define the new classification head
class ClassificationHead(nn.Module):
    def __init__(self, num_features_swin, num_features_densenet):
        super(ClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(num_features_swin + num_features_densenet, 2)  # Assuming binary classification

    def forward(self, x_swin, x_densenet):
        x = torch.cat((x_swin, x_densenet), dim=1)
        x = x.unsqueeze(2).unsqueeze(3)  # Add channel and spatial dimensions
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# Initialize the new classification head
num_features_swin = 1000
num_features_densenet = 1000
# num_features_total = num_features_swin + num_features_densenet
classification_head = ClassificationHead(num_features_swin,num_features_densenet)
classification_head.to(device)

# Define the optimizer and criterion
optimizer = Adam(classification_head.parameters(), lr=0.0001, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()  # Assuming binary classification

# Training loop
num_epochs = 40
patience = 10
min_delta = 0.001
train_losses = []
val_losses = []
best_val_loss = float('inf')
no_improvement_count = 0

for epoch in range(num_epochs):
    model_swin.train()
    model_densenet.train()
    classification_head.train()
    train_loss = 0
    all_train_preds = []
    all_train_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs_swin = model_swin(images)
        # print(outputs_swin.logits.shape)
        
        outputs_densenet = model_densenet(images)
        # print(outputs_densenet.shape)
        combined_features = classification_head(outputs_swin.logits, outputs_densenet)
        optimizer.zero_grad()
        loss = criterion(combined_features, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted_label = combined_features.argmax(-1)
        all_train_preds.extend(predicted_label.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    train_losses.append(train_loss / len(train_loader))
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)
    print(f'Epoch {epoch + 1}/{num_epochs}, train Loss: {train_loss:.4f}, train Acc: {train_accuracy:.4f}')

    # Validation
    model_swin.eval()
    model_densenet.eval()
    classification_head.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for images_val, labels_val in val_loader:
            images_val, labels_val = images_val.to(device), labels_val.to(device)
            outputs_swin_val = model_swin(images_val)
            outputs_densenet_val = model_densenet(images_val)
            combined_features_val = classification_head(outputs_swin_val.logits, outputs_densenet_val)
            loss_val = criterion(combined_features_val, labels_val)
            val_loss += loss_val.item()
            predicted_label = combined_features_val.argmax(-1)
            all_val_preds.extend(predicted_label.cpu().numpy())
            all_val_labels.extend(labels_val.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Check for early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f'Validation loss did not improve for {patience} epochs. Stopping training.')
            break

import matplotlib.pyplot as plt
# Plotting the training and validation losses
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss on Ensemble Model')
plt.legend()
plt.show()


# Load the dataset
test_dataset = datasets.ImageFolder('FLAME_Dataset/Test', transform=transform)

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set the model to evaluation mode
model_swin.eval()
model_densenet.eval()
classification_head.eval()
# Initialize lists to store true labels and predicted probabilities
all_labels = []
all_preds = []
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, auc
import numpy as np
# Iterate over the test dataset and collect true labels and predicted probabilities
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs_swin_test = model_swin(inputs)
        outputs_densenet_test = model_densenet(inputs)
        combined_features_test = classification_head(outputs_swin_test.logits, outputs_densenet_test)
        predicted_label = combined_features_test.argmax(-1)
        all_preds.extend(predicted_label.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())


len(all_preds)
len(all_labels)
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

