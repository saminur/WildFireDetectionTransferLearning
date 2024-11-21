# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:06:05 2024

@author: samin
"""

import torch
from torchvision import models, transforms, datasets
from sklearn.decomposition import PCA
from PIL import Image
import os

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the final classification layer

# Define a transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



# Create datasets
train_dataset = datasets.ImageFolder('FLAME_Dataset/Training', transform=transform)
test_dataset = datasets.ImageFolder('FLAME_Dataset/Test', transform=transform)

# Function to extract features from an image
def extract_features(image_path):
    img = transform(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        features = model(img).squeeze().numpy()
    return features

# Example code to extract features from all images in the training and test datasets
train_features = [extract_features(os.path.join(train_data_path, class_folder, image_name))
                   for class_folder, _, images in os.walk(train_data_path)
                   for image_name in images]
test_features = [extract_features(os.path.join(test_data_path, class_folder, image_name))
                  for class_folder, _, images in os.walk(test_data_path)
                  for image_name in images]

# Flatten the features and apply PCA
train_features_flat = [f.flatten() for f in train_features]
test_features_flat = [f.flatten() for f in test_features]

pca = PCA(n_components=50)  # Choose the number of components based on your needs
train_features_pca = pca.fit_transform(train_features_flat)
test_features_pca = pca.transform(test_features_flat)

# Calculate the variance of each feature in the reduced space
train_variances = train_features_pca.var(axis=0)
test_variances = test_features_pca.var(axis=0)

# Compare the variances between the training and test datasets
# You can use statistical tests or visual inspection here

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Calculate the variance of each feature in the reduced space
train_variances = train_features_pca.var(axis=0)
test_variances = test_features_pca.var(axis=0)

# Statistical tests
t_stat, t_p_value = stats.ttest_ind(train_variances, test_variances)
u_stat, u_p_value = stats.mannwhitneyu(train_variances, test_variances)

print(f"t-test: t-statistic={t_stat}, p-value={t_p_value}")
print(f"Mann-Whitney U test: U-statistic={u_stat}, p-value={u_p_value}")

# Visual inspection
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot([train_variances, test_variances], labels=['Train', 'Test'])
plt.ylabel('Feature Variance')
plt.title('Boxplot of Feature Variances')

plt.subplot(1, 2, 2)
plt.hist(train_variances, alpha=0.5, label='Train', bins=20)
plt.hist(test_variances, alpha=0.5, label='Test', bins=20)
plt.xlabel('Feature Variance')
plt.ylabel('Frequency')
plt.title('Histogram of Feature Variances')
plt.legend()

plt.tight_layout()
plt.show()

