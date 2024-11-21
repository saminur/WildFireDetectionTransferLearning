import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from transformers import ViTForImageClassification, Swinv2ForImageClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, auc

# Define a function to perform majority voting
def majority_vote(predictions, type="soft"):
    if type == "soft":
        # Sum up probabilities from all models
        #print(predictions)
        combined_probs = torch.stack(predictions).mean(dim=0)
        # Get the index of the class with the highest average probability
        predicted_class = torch.argmax(combined_probs, dim=1)
        #print(predicted_class, combined_probs)
        return predicted_class
    elif type == "hard":
        # Sum up probabilities from all models
        #print(predictions, torch.argmax(predictions[0], dim=1))
        combined_probs = torch.stack([torch.argmax(prediction, dim=1) for prediction in predictions])
        # Get the index of the class with the highest average probability
        predicted_class = torch.mode(combined_probs, dim=0).values
        #print(predicted_class, combined_probs)
        return predicted_class



# Define transformations for preprocessing
transform_1 = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

transform_2 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resnet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


# Define the model architecture
model_list = list()

model_1 = models.inception_v3(pretrained=False)
num_classes = 2  
num_features = model_1.fc.in_features
model_1.fc = nn.Linear(num_features, num_classes)

# Load the saved model state dictionary
model_1_path = 'models/inceptionv3_flame2_49.pth'
model_1.load_state_dict(torch.load(model_1_path))


model_2 = models.resnet50(pretrained=False)
num_features = model_2.fc.in_features
model_2.fc = nn.Linear(num_features, num_classes)
model_2_path = 'models/resnet/resnet_flame2_10.pth'
model_2.load_state_dict(torch.load(model_2_path))


model_3_path = 'models/ViT/ViT_FLAME_8'
model_3 = ViTForImageClassification.from_pretrained(model_3_path)

model_4_path = 'models/SwinT/swin_tiny_flame_256_full_StopCri.pth'
model_4 = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
state_dict = torch.load(model_4_path, map_location='cpu')
model_4.load_state_dict(state_dict, strict=False)


model_list.append(('inceptionV3', model_1))
model_list.append(('resnet', model_2))
model_list.append(('vision_transformer', model_3))
model_list.append(('swin_transformer', model_4))

print("Models Used for Voting: \n" , (model[0] for model in model_list))

dataset_path = 'FLAME_Dataset/Test'
# Load the dataset
test_dataset_1 = datasets.ImageFolder(dataset_path, transform=transform_1) # For 299x299 images
# Create data loaders
test_loader_1 = DataLoader(test_dataset_1, batch_size=32, shuffle=False)


# Load the dataset
test_dataset_2 = datasets.ImageFolder(dataset_path, transform=transform_2) # For 224x224 images
# Create data loaders
test_loader_2 = DataLoader(test_dataset_2, batch_size=32, shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for model in model_list:
    model[1].eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for (images_1, labels), (images_2, _) in zip(test_loader_1, test_loader_2):
        #images, labels = images.to(device), labels.to(device)
        predictions = []
        outputs = model_list[0][1](images_1)
        predictions.append(outputs)  

        #outputs = model_list[1][1](images_2)
        #predictions.append(outputs)

        outputs = model_list[2][1](images_2).logits
        predictions.append(outputs) 
 
        outputs = model_list[3][1](images_2).logits
        predictions.append(outputs)   
        
        predicted_class = majority_vote(predictions, "hard")
        all_preds.extend(predicted_class)
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
plt.savefig('results/ensemble/confusion_matrix.png')

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
plt.savefig('results/ensemble/roc_curve.png')