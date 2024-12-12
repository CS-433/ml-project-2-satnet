"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from SatDataset import SatDataset

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16
    EPOCHS = 5
    DATA_PATH = "../dataset/short_augmented_training"
    MODEL_SAVE_PATH = "../models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = SatDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
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

def calculate_accuracy(y_pred, mask):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5) # Apply threshold to convert to binary
    y_pred = y_pred.cpu().numpy().flatten()
    mask = mask.cpu().numpy().flatten()
    return accuracy_score(mask, y_pred)

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16
    EPOCHS = 5
    DATA_PATH = "../dataset/short_augmented_training"
    MODEL_SAVE_PATH = "../models/unet.pth"
    METRICS_SAVE_PATH = "../models/metrics.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = SatDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
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
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            metrics["train_loss"].append(loss.item())

            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(y_pred, mask)
            metrics["train_accuracy"].append(accuracy)

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()
                metrics["val_loss"].append(loss.item())

                accuracy = calculate_accuracy(y_pred, mask)
                metrics["val_accuracy"].append(accuracy)

        val_loss = val_running_loss / (idx + 1)

        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f)

    print("done")