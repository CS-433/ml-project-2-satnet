import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import json
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from utils.unet import UNet
from utils.SatDataset import SatDataset
from utils.helpers import calculate_metrics

if __name__ == "__main__":
    DATA_PATH_TRAINING = "dataset/augmented_dataset/training"
    DATA_PATH_VALIDATION = "dataset/augmented_dataset/validation"
    METRICS_SAVE_PATH = "models/metrics_final.json"
    MODEL_SAVE_PATH = "models/final_unet.pth"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    WEIGHT_DECAY = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    train_dataset = SatDataset(DATA_PATH_TRAINING, training=True)
    val_dataset = SatDataset(DATA_PATH_VALIDATION, training=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": []
    }
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_accuracy = 0
        train_running_f1 = 0
        for id, img_and_grt in enumerate(tqdm(train_dataloader)):
            img = img_and_grt[0].float().to(device)
            grt = img_and_grt[1].float().to(device)
            y_pred = model(img)
            optimizer.zero_grad()
            loss = criterion(y_pred, grt)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            accuracy, f1 = calculate_metrics(y_pred, grt)
            train_running_accuracy += accuracy
            train_running_f1 += f1
        train_loss = train_running_loss / (id + 1)
        train_accuracy = train_running_accuracy / (id + 1)
        train_f1 = train_running_f1 / (id + 1)
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["train_f1"].append(train_f1)

        model.eval()
        val_running_loss = 0
        val_running_accuracy = 0
        val_running_f1 = 0
        with torch.no_grad():
            for id, img_and_grt in enumerate(tqdm(val_dataloader)):
                img = img_and_grt[0].float().to(device)
                grt = img_and_grt[1].float().to(device)
                y_pred = model(img)
                loss = criterion(y_pred, grt)
                val_running_loss += loss.item()
                accuracy, f1 = calculate_metrics(y_pred, grt)
                val_running_accuracy += accuracy
                val_running_f1 += f1
        val_loss = val_running_loss / (id + 1)
        val_accuracy = val_running_accuracy / (id + 1)
        val_f1 = val_running_f1 / (id + 1)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["val_f1"].append(val_f1)

        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Train Accuracy EPOCH {epoch + 1}: {train_accuracy:.4f}")
        print(f"Valid Accuracy EPOCH {epoch + 1}: {val_accuracy:.4f}")
        print(f"Train F1 EPOCH {epoch + 1}: {train_f1:.4f}")
        print(f"Valid F1 EPOCH {epoch + 1}: {val_f1:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f)
    print("done")