import os
import datetime
import sys
import warnings
import logging
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from torchvision.models import densenet201, DenseNet201_Weights
from sklearn.metrics import f1_score
from matplotlib import MatplotlibDeprecationWarning

from data_transform import augmentation_transform, base_transform

warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert str(device) == 'cuda', 'CUDA is not available'

data_path = "data/ukraine-ml-bootcamp-2023/subcat"
USE_SMALL_CLASSES = True
SMALL_CLASS_MAX = True

dataset = ImageFolder(data_path, transform=base_transform)
num_classes = len(dataset.classes)

reversed_groups = {}
for i, classname in enumerate(dataset.classes):
    reversed_groups[i] = int(classname.split('_')[0])
reversed_groups_values = np.array(list(reversed_groups.values()))

small_num_classes = len(set(reversed_groups.values()))
print(f'Small num classes: {small_num_classes}\n Num classes: {num_classes}')

indexes = []
for i in range(small_num_classes):
    indexes.append(np.where(reversed_groups_values == i)[0][0])
indexes.append(small_num_classes)

data = None
start_lr = 0.0001
eval_every = 15
batch_size = 16
epochs = 40
stop_patience = 10
no_improvement = 0


class CustomClassifier(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=128, dropout_rate=0.5):
        super(CustomClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)

        return x


def load_model(from_scratch=False, weights_path=None, pickle_path=None):
    if pickle_path is not None:
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
        return model
    model = densenet201(weights=DenseNet201_Weights.DEFAULT)
    model.classifier = CustomClassifier(model.classifier.in_features, num_classes)
    model = model.to(device)
    if not from_scratch:
        assert weights_path is not None, 'Weights path is None'
        state_dict = torch.load(weights_path)

        # change fc to fc1 in state dict because I was inconsistent with the names of the layers
        if 'classifier.fc.weight' in state_dict:
            state_dict['classifier.fc1.weight'] = state_dict['classifier.fc.weight']
            state_dict['classifier.fc1.bias'] = state_dict['classifier.fc.bias']
            del state_dict['classifier.fc.weight']
            del state_dict['classifier.fc.bias']
        model.load_state_dict(state_dict)
    return model


def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    count = 0
    small_targets = []
    small_outputs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            small_targets.extend(reversed_groups_values[target.cpu()])
            small_outputs.extend(reversed_groups_values[output.argmax(1).cpu()])
            test_loss += criterion(output, target)
            count += 1

    test_loss /= count
    f1 = f1_score(small_targets, small_outputs, average='macro')
    return test_loss, f1


def plot_train_val(plot_data):
    plot_data_array = np.array(plot_data)
    train_losses = plot_data_array[:, 0]
    val_losses = plot_data_array[:, 1]
    val_f1s = plot_data_array[:, 2]

    # Calculate iterations for x-axis (index * 100)
    iterations = [idx * eval_every * batch_size for idx in range(len(plot_data))]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, val_losses, marker='o', color='green', label='Validation Loss')
    plt.plot(iterations, train_losses, marker='o', color='blue', label='Train Loss')
    plt.plot(iterations, val_f1s, marker='o', color='red', label='Validation F1')
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
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.targets is not None:
            target = self.targets[idx]
            return image, target

        return image


if __name__ == '__main__':
    print("Loading data...")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = augmentation_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Loading model...")
    model = load_model(from_scratch=True)

    # Only keep the classifier layers trainable
    for name, param in model.named_parameters():
        param.requires_grad = True
    print('Model loaded')

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    today = datetime.datetime.now().strftime("%m-%d_%H:%M")
    run_folder = f"runs/train-{today}"
    os.makedirs(run_folder)

    # Train
    print("Training...")
    best_f1 = 0
    plot_data = []
    total_loss = 0
    total_count = 0

    description = f"lr={start_lr}, batch_size={batch_size}, epochs={epochs}, stop_patience={stop_patience}\n" \
                  f"classification layer: {model.classifier}\n" \
                  f"scheduler: patience {scheduler.patience} factor {scheduler.factor}\n" \
                  f"model: densenet-201" \
                  f"NOTE: added more classes\n"

    log_filename = os.path.join(run_folder, 'log.txt')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    stdout_logger = logging.getLogger('stdout')
    stdout_logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    stdout_logger.addHandler(stdout_handler)

    with open(os.path.join(run_folder, 'description.txt'), 'w') as f:
        f.write(description)

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            with torch.no_grad():
                train_loss = loss_fn(output, target)

            if USE_SMALL_CLASSES:
                small_target = reversed_groups_values[target.cpu()]
                small_target = torch.from_numpy(small_target).to(device).float()
                if SMALL_CLASS_MAX:
                    small_output = reversed_groups_values[output.argmax(1).cpu()]
                    small_output = torch.from_numpy(small_output).to(device).float()
                else:
                    small_output = torch.zeros(len(indexes) - 1, output.shape[0]).to(device)
                    for i in range(len(indexes) - 1):
                        start_idx = indexes[i]
                        end_idx = indexes[i + 1]
                        small_output[i] = output[:, start_idx:end_idx].sum(dim=1)
                    small_output = small_output.transpose(1, 0)
                small_output.requires_grad = True
                loss = loss_fn(output, target) + loss_fn(small_output, small_target)
            else:
                loss = loss_fn(output, target)
            total_loss += train_loss.item()
            total_count += 1
            loss.backward()
            optimizer.step()
            if batch_idx % eval_every == 0:
                val_loss, val_f1 = test(model, test_loader)
                model.train()
                scheduler.step(val_loss)
                log_text = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain loss: {:.6f}\tVal loss: {:.6f}\tVal F1: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                           total_loss / total_count, val_loss, val_f1)
                logging.info(log_text)
                print(log_text)
                total_loss = 0
                total_count = 0

                # save plot
                plot_data.append((train_loss.item(), val_loss.item(), val_f1))
                if len(plot_data) > 1:
                    plt = plot_train_val(plot_data)
                    plot_path = os.path.join(run_folder, 'plot.png')

                    plt.savefig(plot_path)
                    plt.close()

                # save model
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    with open(os.path.join(run_folder, 'model.pkl'), 'wb') as f:
                        pickle.dump(model, f)
                    logging.info('Model saved')
                    print('Model saved')
                    no_improvement = 0
        no_improvement += 1
        if no_improvement >= stop_patience:
            logging.info('Early stopping')
            print('Early stopping')
            break
