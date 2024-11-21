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
from torch.utils.data import DataLoader

# Load the Swin Transformer model and image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_features = model.config.hidden_size

# Freeze all parameters in the model
for param in model.parameters():
    param.requires_grad = False
    
# Replace the classifier head
# model.classifier = nn.Sequential(
#     nn.Linear(num_features, 32),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(32, 2)
# )
model.classifier = nn.Sequential(
    nn.Linear(num_features, 32),
    nn.ReLU(),
    # nn.Dropout(0.5),
    nn.Linear(32, 2)
)

    
# Optional: Fine-tune the entire model or just the new head
optimizer = Adam(model.parameters(), lr=0.0001,weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                        save_as_images, display_in_terminal)

# Load pre-trained BigGAN model
# model_gan = BigGAN.from_pretrained('biggan-deep-256')
# model_gan.to(device)

# Define GAN-based augmentation function using BigGAN
# def biggan_augmentation(image, truncation=0.4):
#     # Generate a random noise vector
#     noise = truncated_noise_sample(truncation=truncation, batch_size=1, seed=None)
#     noise = torch.tensor(noise, dtype=torch.float32)
    
#     # Generate an image using BigGAN
#     with torch.no_grad():
#         # Convert the one-hot vector to a PyTorch tensor
#         one_hot = torch.tensor(one_hot_from_names(['red panda'], batch_size=1))
        
#         # Ensure that both inputs are on the same device (e.g., GPU or CPU)
#         noise = noise.to(device)
#         one_hot = one_hot.to(device)
        
#         output = model_gan(noise, one_hot, truncation=truncation)
#     output = output[:, :3, :, :]  # Keep only RGB channels
    
#     # Convert the generated image tensor to a PIL Image
#     fake_image_tensor = transforms.ToPILImage()(output[0].cpu())
    
    
#     fake_image_tensor = transforms.ToTensor()(fake_image_tensor)
   
    
   
#     # Convert the original image to a PyTorch Tensor
#     image_tensor = transforms.ToTensor()(image)
    
#     # Resize the fake image tensor to match the shape of the original image tensor
#     fake_image_tensor = torch.nn.functional.interpolate(fake_image_tensor.unsqueeze(0), size=image_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
   
#     # Combine the fake image with the original image
#     augmented_image_tensor = torch.add(image_tensor, fake_image_tensor)
   
#     # Convert back to PIL Image
#     augmented_image = transforms.ToPILImage()(augmented_image_tensor)
   
#     return augmented_image


# Use the augmentation function in your data loader or training loop


# Define a simple GAN model
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

# Define transformations including GAN-based augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda x: gan_augmentation(x)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])


# Load the dataset
dataset = datasets.ImageFolder('FLAME_Dataset/Training', transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# Split the dataset into training and validation sets
# train_size = int(0.6 * len(dataset))
# val_size = int(0.1 * len(dataset))
# test_size = len(dataset) - (train_size + val_size)
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_epochs = 100
patience = 10  # Number of epochs with no improvement after which training will be stopped
min_delta = 0.001  # Minimum change in validation loss to be considered as improvement

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
    print(f'Epoch {epoch + 1}/{num_epochs}, train Loss: {loss.item():.4f}, train Acc: {train_accuracy:.4f}')

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
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_val.item():.4f}, Val Acc: {val_accuracy:.4f}')

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
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig(path_plot+'validation_loss_curve_64_32.png')

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda x: gan_augmentation(x)),
    transforms.ToTensor(),
])


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
# Iterate over the test dataset and collect true labels and predicted probabilities
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        logits_test = model(inputs).logits
        predicted_label = logits_test.argmax(-1)
        all_preds.extend(predicted_label.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())


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


path_model = 'swin_tiny_flame_256_full.pth'
torch.save(model.state_dict(),path_model)


