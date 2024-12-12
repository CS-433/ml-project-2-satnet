"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import json
from unet import UNet
from SatDataset import SatDataset
from helpers import calculate_accuracy
if __name__ == "__main__":
    # Paths to the data and to save the model and metrics
    DATA_PATH = "../dataset/short_augmented_training"
    METRICS_SAVE_PATH = "../models/metrics.json"
    MODEL_SAVE_PATH = "../models/unet.pth"
    # Hyperparameters
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    WEIGHT_DECAY = 1e-2
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the dataset
    dataset = SatDataset(DATA_PATH)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    metrics={
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for id, img_and_grt in enumerate(tqdm(train_dataloader)):
            img = img_and_grt[0].float().to(device)
            grt = img_and_grt[1].float().to(device)
            y_pred = model(img)
            optimizer.zero_grad()
            loss = criterion(y_pred, grt)
            train_running_loss+=loss.item()
            metrics["train_loss"].append(loss.item())
            loss.backward()
            optimizer.step()
            accuracy = calculate_accuracy(y_pred,grt)
            metrics["train_accuracy"].append(accuracy)
        train_loss = train_running_loss/(id+1)
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for id, img_and_grt in enumerate(tqdm(val_dataloader)):
                img = img_and_grt[0].float().to(device)
                grt = img_and_grt[1].float().to(device)
                y_pred = model(img)
                loss = criterion(y_pred,grt)
                metrics["val_loss"].append(loss.item())
                val_running_loss += loss.item()
                metrics["val_loss"].append(loss.item())
                accuracy = calculate_accuracy(y_pred,grt)
                metrics["val_accuracy"].append(accuracy)
        val_loss = val_running_loss / (id+1)
        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f)
    print("done")
    """
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import json
from unet import UNet
from SatDataset import SatDataset
from helpers import calculate_accuracy

if __name__ == "__main__":
    # Paths to the data and to save the model and metrics
    DATA_PATH = "../dataset/mahmoud_training"
    METRICS_SAVE_PATH = "../models/metrics.json"
    MODEL_SAVE_PATH = "../models/unet.pth"
    # Hyperparameters
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    WEIGHT_DECAY = 1e-2
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    # Load the dataset
    dataset = SatDataset(DATA_PATH)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_accuracy = 0
        for id, img_and_grt in enumerate(tqdm(train_dataloader)):
            img = img_and_grt[0].float().to(device)
            grt = img_and_grt[1].float().to(device)
            y_pred = model(img)
            optimizer.zero_grad()
            loss = criterion(y_pred, grt)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            accuracy = calculate_accuracy(y_pred, grt)
            train_running_accuracy += accuracy
        train_loss = train_running_loss / (id + 1)
        train_accuracy = train_running_accuracy / (id + 1)
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)

        model.eval()
        val_running_loss = 0
        val_running_accuracy = 0
        with torch.no_grad():
            for id, img_and_grt in enumerate(tqdm(val_dataloader)):
                img = img_and_grt[0].float().to(device)
                grt = img_and_grt[1].float().to(device)
                y_pred = model(img)
                loss = criterion(y_pred, grt)
                val_running_loss += loss.item()
                accuracy = calculate_accuracy(y_pred, grt)
                val_running_accuracy += accuracy
        val_loss = val_running_loss / (id + 1)
        val_accuracy = val_running_accuracy / (id + 1)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)

        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f)
    print("done")