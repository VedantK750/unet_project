import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet
from loading_dataset import MulticlassSegmentation


# Define color-to-class mapping (i took this from kaggle)
color_to_class = {
    (155, 38, 182): 0,  # obstacles
    (14, 135, 204): 1,  # water
    (124, 252, 0): 2,   # nature
    (255, 20, 147): 3,  # moving
    (169, 169, 169): 4  # landable
}

image_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MulticlassSegmentation(
    image_dir="/home/krish/Desktop/unet/classes_dataset/original_images",
    label_dir="/home/krish/Desktop/unet/classes_dataset/label_images_semantic",
    color_to_class=color_to_class,
    transform=image_transform,
    augmentation=True
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

test_dataset.dataset.augmentation = False


train_loader = dataloader.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = dataloader.DataLoader(test_dataset, batch_size=8, shuffle=False)


learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
num_epochs = 10
num_workers = 2
image_height = 512
image_width = 512
pin_memory = True
load_model = True
loss_fn = torch.nn.CrossEntropyLoss()
model = UNet(in_channels=3, num_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()


def train_fn(loader, model, optimizer, loss_fn, scaler,device):
    model.train()
    loop = tqdm(loader)
    running_loss = 0 
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

# Validation function to check accuracy
def evaluate_fn(loader, model, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            _, predicted = torch.max(predictions, 1)  # Get the class with highest score

            correct += (predicted == targets).sum().item()
            total += targets.numel()  # total number of pixels

    accuracy = correct / total
    return accuracy

# Inference and visualization function
def visualize_inference(loader, model, device, num_images=5):
    model.eval()
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)

    predictions = model(images)
    _, predicted = torch.max(predictions, 1)  # Get the predicted class

    # Show original images, targets, and predictions
    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Original image
        axs[0].imshow(images[i].cpu().permute(1, 2, 0))  # Convert from CHW to HWC for visualization
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Ground truth
        axs[1].imshow(targets[i].cpu(), cmap='tab10')  # Show target using the color map
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')

        # Predicted segmentation map
        axs[2].imshow(predicted[i].cpu(), cmap='tab10')  # Show predicted output
        axs[2].set_title("Prediction")
        axs[2].axis('off')

        plt.show()

# Training loop with evaluation and visualization
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
    
    # Evaluate on the test set and print accuracy
    accuracy = evaluate_fn(test_loader, model, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Visualize inference for the first few images in the test set
    visualize_inference(test_loader, model, device, num_images=3)


