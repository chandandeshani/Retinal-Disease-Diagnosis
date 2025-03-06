import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix



#  Detect GPU or fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Using device: {DEVICE} ")
if DEVICE == "cuda":
    print(f" Running on: {torch.cuda.get_device_name(0)}\n")





#  Define transformations with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images for Swin Transformer
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomRotation(10),  # Augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.18376897, 0.18376897, 0.18376897], std=[0.18822514, 0.18822514, 0.18822514])  # Updated normalization
])

#  Load Dataset
data_dir = "OCT"
train_dir = f"{data_dir}/train"
test_dir = f"{data_dir}/test"

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

#  Find class indices
class_indices = {class_name: [] for class_name in train_dataset.classes}
for idx, (_, label) in enumerate(train_dataset.samples):
    class_indices[train_dataset.classes[label]].append(idx)

#  Undersampling: Select 8616 images per class
min_samples = 8616
balanced_indices = []
for class_name, indices in class_indices.items():
    selected_indices = random.sample(indices, min_samples)  # Undersample
    balanced_indices.extend(selected_indices)

#  Create DataLoader with Balanced Subset
train_sampler = SubsetRandomSampler(balanced_indices)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

#  Load Pretrained Swin Transformer Model
model = models.swin_v2_t(weights="DEFAULT")  # Load Swin Transformer
num_ftrs = model.head.in_features  # Get the number of input features in the last layer
model.head = nn.Linear(num_ftrs, len(train_dataset.classes))  # Replace final layer

#  Move model to GPU (if available)
model = model.to(DEVICE)
print(f" Model is on: {next(model.parameters()).device}")

#  Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Use AdamW with weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR after every 5 epochs

#  Training Loop
def train_model(model, train_loader, epochs=15):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)  #  Move data to GPU
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            #  Print every 10 batches to see training progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_acc = correct / total
        print(f" Epoch {epoch+1}/{epochs} Completed - Avg Loss: {running_loss/len(train_loader):.4f} - Accuracy: {epoch_acc:.4f}")
        scheduler.step()  # Adjust learning rate

#  Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)  #  Move data to GPU
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #  Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    cm = confusion_matrix(all_labels, all_preds)

    #  Print Metrics
    print(f"\n Accuracy: {acc:.4f}")
    print(f" Precision: {precision:.4f}")

    #  Print Confusion Matrix (Raw)
    print("\n Confusion Matrix:")
    print(cm)  # Print matrix in terminal

#  Train and Evaluate Model
train_model(model, train_loader, epochs=15)
evaluate_model(model, test_loader)

#  Save the trained model
model_path = "swin_transformer_retinal_disease.pkl"
torch.save(model.state_dict(), model_path)
print(f"Model saved as {model_path}")
