import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time


# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the dataset
dataset = datasets.ImageFolder('FLAME_Dataset/Training', transform=transform)
train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2], generator = torch.Generator().manual_seed(42))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Load pre-trained InceptionV3 model
model = models.inception_v3(pretrained=True)
#model.aux_logits = False

# Freeze all parameters in the model
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier to match the number of classes in the dataset
num_classes = 2
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion_aux = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Track train and validation loss
train_losses = []
validation_losses = []

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    # Assume training loop starts here
    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, aux_outputs = model(images)
        loss_main = criterion(outputs, labels)
	# Compute auxiliary losses
        loss_aux = criterion_aux(aux_outputs, labels)

        # Total loss
        #loss = loss_main + 0.3 * loss_aux
        loss = loss_main

        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * images.size(0)  # Track loss for each batch
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate average training loss for the epoch 
    train_losses.append(epoch_train_loss)
    train_accuracy = correct / total
    # Training loop ends here
    end_time = time.time()

    # Compute the time taken for the epoch in seconds
    epoch_time_seconds = end_time - start_time

    # Convert the time into hours, minutes, and seconds
    hours, remainder = divmod(epoch_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time taken for epoch: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
    
    # Validation
    model.eval()
    epoch_validation_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_validation_loss += loss.item() * images.size(0)  # Track loss for each batch
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average validation loss for the epoch
    validation_losses.append(epoch_validation_loss)
    validation_accuracy= accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f'Accuracy: {validation_accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Confusion Matrix:\n{cm}')
    
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
    #plt.show()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {100 * train_accuracy:.2f}%, '
          f'Validation Loss: {epoch_validation_loss:.4f}, Validation Acc: {100 * validation_accuracy:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), f'models/inceptionv3_flame2_{epoch+1}.pth')



# Plot the training and validation losses
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train')
plt.plot(range(1, num_epochs+1), validation_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
plt.savefig('results/train_vs_validation_loss.png')

