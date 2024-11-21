# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 00:47:30 2024

@author: samin
"""

from itertools import product
from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import densenet201
from sklearn.metrics import accuracy_score

# Define hyperparameters to tune
learning_rates = [0.0001, 0.001, 0.01]
weight_decays = [0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64]
num_epochs = [30, 40, 50]

# Define data augmentation techniques
data_augmentation = [
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomGrayscale(p=0.1),
]

# Define optimization techniques
optimizers = [Adam, torch.optim.SGD]

# Define regularization techniques
regularizations = [nn.Dropout(0.2), nn.Dropout(0.3), nn.Dropout(0.4)]

# Create a grid of hyperparameters to search over
param_grid = product(learning_rates, weight_decays, batch_sizes, num_epochs, data_augmentation, optimizers, regularizations)

best_accuracy = 0.0
best_params = None

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

def evaluate_model_on_test_data(model_swin, model_densenet, class_head, test_loader):
    # Set the model to evaluation mode
    model_swin.eval()
    model_densenet.eval()
    class_head.eval()

    # Initialize lists to store true labels and predicted probabilities
    all_labels = []
    all_preds = []

    # Iterate over the test dataset and collect true labels and predicted probabilities
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_swin_test = model_swin(inputs)
            outputs_densenet_test = model_densenet(inputs)
            combined_features_test = class_head(outputs_swin_test.logits, outputs_densenet_test)
            predicted_label = combined_features_test.argmax(-1)
            all_preds.extend(predicted_label.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    # precision = precision_score(all_labels, all_preds)
    # recall = recall_score(all_labels, all_preds)
    # f1 = f1_score(all_labels, all_preds)

    return accuracy

# # Evaluate the model on the test dataset
# test_accuracy = evaluate_model_on_test_data(model_swin, model_densenet, class_head, test_loader)
# print(f'Hyperparameters: lr={lr}, wd={wd}, bs={bs}, ne={ne}, da={da}, opt={opt}, reg={reg}')
# print(f'Test Accuracy: {test_accuracy}')


patience = 10
min_delta = 0.001    
for lr, wd, bs, ne, da, opt, reg in param_grid:
    # Load the dataset
    transform = transforms.Compose([
        da,
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder('FLAME_Dataset/Training', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

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
    class_head = ClassificationHead(1000, 1000)
    class_head.to(device)

    # Define the optimizer and criterion
    optimizer = opt(class_head.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improvement_count = 0

    # Training loop
    for epoch in range(ne):
        model_swin.train()
        model_densenet.train()
        class_head.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
    
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs_swin = model_swin(images)
            outputs_densenet = model_densenet(images)
            combined_features = class_head(outputs_swin.logits, outputs_densenet)
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
        print(f'Epoch {epoch + 1}/{ne}, train Loss: {train_loss:.4f}, train Acc: {train_accuracy:.4f}')
    
        # Validation
        model_swin.eval()
        model_densenet.eval()
        class_head.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
    
        with torch.no_grad():
            for images_val, labels_val in val_loader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                outputs_swin_val = model_swin(images_val)
                outputs_densenet_val = model_densenet(images_val)
                combined_features_val = class_head(outputs_swin_val.logits, outputs_densenet_val)
                loss_val = criterion(combined_features_val, labels_val)
                val_loss += loss_val.item()
                predicted_label = combined_features_val.argmax(-1)
                all_val_preds.extend(predicted_label.cpu().numpy())
                all_val_labels.extend(labels_val.cpu().numpy())
    
        val_losses.append(val_loss / len(val_loader))
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        print(f'Epoch {epoch + 1}/{ne}, Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Stopping training.')
                break


    # Load the dataset
    test_dataset = datasets.ImageFolder('FLAME_Dataset/Test', transform=transform)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Evaluate the model on the test dataset
    test_accuracy = evaluate_model_on_test_data(model_swin, model_densenet, class_head, test_loader)
    print(f'Hyperparameters: lr={lr}, wd={wd}, bs={bs}, ne={ne}, da={da}, opt={opt}, reg={reg}')
    print(f'Test Accuracy: {test_accuracy}')

    # Update best_params if current accuracy is better
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_params = (lr, wd, bs, ne, da, opt, reg)

print(f'Best hyperparameters: {best_params}')
