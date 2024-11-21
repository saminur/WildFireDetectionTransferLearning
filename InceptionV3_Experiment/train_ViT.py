import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from torchvision.models import vision_transformer
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time


# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(224),
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
labels = ["Fire", "No_Fire"]
model_name_or_path = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
#model = vision_transformer.vit_base_patch16_224(pretrained=True, num_classes=2)  
#config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
#model = ViTModel.from_pretrained('google/vit-base-patch16-224', config=config)
#model.config.num_classes = 2  

# Modify the classifier to match the number of classes in the dataset
num_classes = 2
model.config.num_labels = num_classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track train and validation loss
train_losses = []
validation_losses = []

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 100
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

        outputs = model(images).logits  # The ViTModel returns a tuple, we only need the logits
  
        batch_size = images.size(0)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * images.size(0)  # Track loss for each batch
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate average training loss for the epoch 
    train_losses.append(epoch_train_loss)
    #print(outputs, labels, loss)
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
            outputs = model(images).logits
            #outputs = outputs.view(-1, outputs.shape[-1])
            #loss = criterion(outputs, labels)
            epoch_validation_loss += loss.item() * images.size(0)  # Track loss for each batch
            
            preds = torch.argmax(outputs, dim=1)            
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average validation loss for the epoch
    #print(model(images))
    #print(outputs, preds, labels)
    #print(all_labels, "\n" , all_preds)
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
        
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {100 * train_accuracy:.2f}%, '
          f'Validation Loss: {epoch_validation_loss:.4f}, Validation Acc: {100 * validation_accuracy:.2f}%')

    # Save the trained model
    model.save_pretrained(f'models/ViT/ViT_FLAME_{epoch+1}', from_pt=True)



# Plot the training and validation losses
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train')
plt.plot(range(1, num_epochs+1), validation_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
plt.savefig('results/ViT/train_vs_validation_loss.png')

