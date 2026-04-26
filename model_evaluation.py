#model evaluation on test data and genrate Confusion Matrix
from torchvision import transforms, models
from dataloader import load_test_dataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
import torch
import os

data_root = 'data/'
checkpoint_path = 'resnet_50_train_acc_9986_val_9873.pth'

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
if os.path.exists(data_root):
    test_loader = load_test_dataset(data_root, batch_size)
    
checkpoint_load = torch.load(checkpoint_path, map_location=device)
class_names = checkpoint_load['class_names']
num_classes = len(class_names)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint_load['state_dict'])
model = model.to(device)
model.eval()

# Test evaluation
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Accuracy
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(18,18))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('Confusion Matrix.png')

# Classification report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))
