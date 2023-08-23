import os
import datetime

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.densenet import densenet201, DenseNet201_Weights

device = 'cuda'
pretrained_path = 'densenet201.pth'
data = None
eval_every = 15
batch_size = 32
num_classes = 6
epochs = 20


def load_model(from_scratch=False):
    model = None
    if from_scratch:
        model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        model = model.to(device)
        model.classifier = nn.Sequential(
            nn.Linear(1920, 512),  # Adjust input features based on the model architecture
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output layer with your number of classes
        )
    assert model is not None, 'Model is None'
    for name, param in model.named_parameters():
        if "classifier" in name:  # Keep the classifier layers trainable
            param.requires_grad = True
        else:
            param.requires_grad = False
    print('DenseNet-201 Model loaded')
    return model


def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            count += 1

    test_loss /= count
    return test_loss


def plot_train_val(plot_data):
    train_losses = [item[0] for item in plot_data]
    val_losses = [item[1] for item in plot_data]

    # Calculate iterations for x-axis (index * 100)
    iterations = [idx * 100 for idx in range(len(plot_data))]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, val_losses, marker='o', color='green', label='Validation Loss')
    plt.plot(iterations, train_losses, marker='o', color='blue', label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    return plt


class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.image_paths = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        target = self.targets[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image_tensor = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, target


if __name__ == '__main__':
    print("Loading data...")
    data_path = "data/ukraine-ml-bootcamp-2023/images/train_images"
    train_csv = "data/ukraine-ml-bootcamp-2023/train.csv"

    today = datetime.datetime.now().strftime("%m-%d_%H:%M")
    run_folder = f"runs/train-{today}"
    os.makedirs(run_folder)

    csv_data = np.genfromtxt(train_csv, delimiter=',', skip_header=1, dtype=str)

    images = [os.path.join(data_path, image) for image in csv_data[:, 0]]
    targets = csv_data[:, 1].astype(int)

    train_part = int(len(targets) * 0.8)
    train_x, train_y = images[:train_part], targets[:train_part]
    test_x, test_y = images[train_part:], targets[train_part:]

    train_data = CustomDataset(train_x, train_y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data = CustomDataset(test_x, test_y)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    print("Loading model...")
    model = load_model(from_scratch=True)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    print("Training...")
    best_loss = np.inf
    plot_data = []
    total_loss = 0
    total_count = 0
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            total_count += 1
            loss.backward()
            optimizer.step()
            if batch_idx % eval_every == 0:
                val_loss = test(model, test_loader)
                model.train()
                scheduler.step(val_loss)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain loss: {:.6f}\tVal loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    total_loss / total_count, val_loss))
                total_loss = 0
                total_count = 0

                # save plot
                plot_data.append((loss.item(), val_loss))
                if len(plot_data) > 1:
                    plt = plot_train_val(plot_data)
                    plot_path = os.path.join(run_folder, 'plot.png')

                    plt.savefig(plot_path)
                    plt.close()

                # save model
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(run_folder, 'model.pth'))
                    print('Model saved')
