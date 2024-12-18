import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from cnn import SatelliteRoadCNN
from SatDataset import SatDataset
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from helpers import *
import json
import numpy as np

def training_cnn():
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8
    EPOCHS = 10
    WEIGHT_DECAY = 1e-3
    DATA_PATH = 'dataset/training'
    MODEL_SAVE_PATH = "models/cnn_100_batch8.pth"
    METRICS_SAVE_PATH = "models/cnn_100_batch16.json"
    current_path = os.getcwd()
    DATA_PATH = os.path.join(current_path,DATA_PATH)
    MODEL_SAVE_PATH = os.path.join(current_path,MODEL_SAVE_PATH)
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

    model = SatelliteRoadCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.2]).to(device)).to(device)
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        accuracies = []
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            #mask = (mask>=0.5).float().to(device)
            #print(img.shape)
            y_pred = model(img)

            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            mask = mask.squeeze(0).squeeze(0)
            mask=(mask>=0.5).int()
            pred_mask = y_pred.squeeze(0).squeeze(0).cpu().detach()
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask >= 0.5).int().cpu()
            y_pred = pred_mask.numpy().flatten()
            grt = mask.cpu().numpy().flatten()
            accuracy = accuracy_score(grt,y_pred)
            accuracies.append(accuracy)
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        val_running_accuracy = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                pred_mask = y_pred.squeeze(0).squeeze(0).cpu().detach()
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = (pred_mask >= 0.5).int().cpu()
                y_pred = pred_mask.numpy().flatten()
                grt = mask.cpu().numpy().flatten()
                accuracy = accuracy_score(grt,y_pred)
                val_running_loss += loss.item()
                val_running_accuracy += accuracy
                
        val_loss = val_running_loss / (idx + 1)
        val_accuracy = val_running_accuracy / (idx + 1)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(np.mean(accuracies))
        
        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)

    
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f)
    print("done")


if __name__ == "__main__":
    training_cnn()