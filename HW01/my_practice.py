# numerical operations
import math
import numpy as np

# read/write data
import pandas as pd
import os
import csv

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# tqdm
from tqdm import tqdm

# tensorboard
from torch.utils.tensorboard import SummaryWriter


# some utility functions
def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(dataset, train_ratio, seed):
    """Splits the dataset into training and validation sets."""
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_dataset), np.array(valid_dataset)

def predict(test_loader, model, device):
    model.eval()
    preds = []
    for data in test_loader:
        with torch.no_grad():
            inputs = data.to(device)
            outputs = model(inputs)
            preds.append(outputs.detach().cpu().numpy())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

# dataset class
class COVID19Dataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.tensor(y, dtype=torch.float32)
        self.x = torch.tensor(x, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

# model class
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

# select features
def select_features(train_data, valid_data, test_data, select_all=True):
    y_train = train_data[:, -1]
    y_valid = valid_data[:, -1]

    train_rows, valid_rows, test_rows = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feats = list(range(1, train_rows.shape[1]))
    else:
        feats = [1, 2, 3, 4]
    return train_rows[:, feats], valid_rows[:, feats], test_rows[:, feats], y_train, y_valid

# begin training
def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    writer = SummaryWriter()
    os.makedirs('./models', exist_ok=True)

    # parameters
    n_epochs, best_loss, step, stop_cnt = config['n_epochs'], float('inf'), 0, 0
    loss_record = []

    # training loop
    for epoch in range(n_epochs):
        model.train()
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            step += 1
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})
        mean_loss1 = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_loss1, epoch)

        # validation
        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_record.append(loss.detach().item())

        mean_loss2 = sum(loss_record) / len(loss_record)
        print(f"Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_loss1:.4f}, Valid loss: {mean_loss2:.4f}")
        writer.add_scalar('Loss/valid', mean_loss2, epoch)

        if mean_loss2 < best_loss:
            best_loss = mean_loss2
            stop_cnt = 0
            torch.save(model.state_dict(), config['save_path'])
        else:
            stop_cnt += 1

        if stop_cnt >= config['early_stop']:
            print("Early stopping")
            break

# configurations
device = "cuda" if torch.cuda.is_available() else "cpu"
config = {
    'seed': 20030221,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 3000,  # Number of epochs.
    'batch_size': 256,
    'lr': 1e-5,
    'momentum': 0.9,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './HW01/models/model.ckpt',  # Your model will be saved here.
}

# dataloader
same_seed(config["seed"])

train_data, test_data = (
    pd.read_csv("./HW01/data/covid.train.csv").values,
    pd.read_csv("./HW01/data/covid.test.csv").values,
)

train_data, valid_data = train_valid_split(
    train_data, config["valid_ratio"], config["seed"]
)

# Print out the data size.
print(
    f"""train_data size: {train_data.shape}
    valid_data size: {valid_data.shape}
    test_data size: {test_data.shape}"""
)

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_features(
    train_data, valid_data, test_data, config["select_all"]
)

# Print out the number of features.
print(f"number of features: {x_train.shape[1]}")

train_dataset, valid_dataset, test_dataset = (
    COVID19Dataset(x_train, y_train),
    COVID19Dataset(x_valid, y_valid),
    COVID19Dataset(x_test),
)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
)

# start training
model = My_Model(input_dim=x_train.shape[1]).to(device)
trainer(train_loader, valid_loader, model, config, device)


# start testing
# def save_pred(preds, file):
#     """Save predictions to specified file"""
#     with open(file, "w") as fp:
#         writer = csv.writer(fp)
#         writer.writerow(["id", "tested_positive"])
#         for i, p in enumerate(preds):
#             writer.writerow([i, p])


# model = My_Model(input_dim=x_train.shape[1]).to(device)
# model.load_state_dict(torch.load(config["save_path"]))
# preds = predict(test_loader, model, device)
# save_pred(preds, "pred.csv")
