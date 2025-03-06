import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

#  Define Dataset Paths
data_dir = "OCT"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Define Transformations (without Normalization for Mean/Std Calculation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#  Load Datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

#  Data Loaders (Optional: for batch processing if needed)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

#  Dataset Summary
print("\n Dataset Summary")
print("-------------------------")
print(f" Number of Classes: {len(train_dataset.classes)}")
print(f" Classes: {train_dataset.classes}")
print(f" Total Training Samples: {len(train_dataset)}")
print(f" Total Testing Samples: {len(test_dataset)}\n")

#  Count Samples per Class
def count_samples(dataset):
    class_counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1
    return class_counts

train_class_counts = count_samples(train_dataset)
test_class_counts = count_samples(test_dataset)

print(" Samples per Class (Train)")
for cls, count in train_class_counts.items():
    print(f"  - {cls}: {count}")

print("\n Samples per Class (Test)")
for cls, count in test_class_counts.items():
    print(f"  - {cls}: {count}")

#  Compute Mean & Standard Deviation
print("\n Computing Dataset Mean & Standard Deviation...")
means, stds = [], []

for img, _ in train_dataset:
    means.append(torch.mean(img, dim=(1, 2)))
    stds.append(torch.std(img, dim=(1, 2)))

dataset_mean = torch.stack(means).mean(dim=0).numpy()
dataset_std = torch.stack(stds).mean(dim=0).numpy()

print(f" Dataset Mean: {dataset_mean}")
print(f" Dataset Std Dev: {dataset_std}\n")

#  Display Sample Images from Each Class
def show_sample_images(dataset, num_samples=4):
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
    class_names = dataset.classes
    shown_classes = set()
    count = 0

    for img, label in dataset:
        if class_names[label] not in shown_classes:
            axes[count].imshow(img.permute(1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
            axes[count].set_title(class_names[label])
            axes[count].axis("off")
            shown_classes.add(class_names[label])
            count += 1
        if count == num_samples:
            break

    plt.show()

print(" Displaying Sample Images from Each Class...")
show_sample_images(train_dataset)

