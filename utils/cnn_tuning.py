import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from torch import optim, nn
from SatDataset import SatDataset
from cnn import SatelliteRoadCNN
from torch.utils.data import DataLoader
from helpers import *
from tqdm import tqdm
from sklearn.metrics import f1_score,confusion_matrix
def tuning(device, root,model_pth):
    model = SatelliteRoadCNN().to(device)
    print("Loading model")
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = SatDataset(root)
    train_dataloader = DataLoader(dataset=image_dataset,
                                  batch_size=1,
                                  shuffle=False)
    treshold = np.arange(0.01,0.05,0.001)
    """
    f1score = []
    for idx, img_mask in enumerate(train_dataloader):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].int().to(device)
        mask = mask.squeeze(0).squeeze(0).cpu()
        pred_mask = model(img)
        pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().detach()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask >= 0.05).int().cpu()
        predictions= np.array(pred_mask).ravel()
        print(pred_mask[mask==1])
        
        mask = np.array(mask).ravel()
        fig = plt.figure()
        
        print(predictions[mask==1])
        fig.add_subplot(1, 1, 1)
        plt.imshow(pred_mask, cmap="gray")
        plt.show()
        print(confusion_matrix(mask, predictions))
    

    for idx, img_mask in enumerate(train_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            mask = mask.squeeze(0).squeeze(0).cpu().numpy()
            pred_mask = model(img)
            pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().detach()
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask >= 0.021).int().cpu()
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)
            plt.imshow(pred_mask, cmap="gray")
            plt.show()
    """
    
    f1scores = []
    for tresh in tqdm(treshold):
        f1score = []
        for idx, img_mask in enumerate(train_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            mask = mask.squeeze(0).squeeze(0)
            
            mask=(mask>=0.5).int()
            pred_mask = model(img)
            pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().detach()
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask >= tresh).int().cpu()
            y_pred = pred_mask.numpy().flatten()
            grt = mask.cpu().numpy().flatten()
            f1 = f1_score(grt,y_pred)
            f1score.append(f1)
        f1scores.append(np.mean(f1score))
    idx = np.argmax(f1scores)  
    print(treshold[idx])
    print(f1scores[idx])  
    
    
            
            
            
if __name__ == "__main__":
    DATA_PATH = "dataset/TrainingInde/test"
    MODEL_PATH = "models/cnn_2000_batch8.pth"
    OUTPUT_DIR = "dataset/short_testing/predicted"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuning(device,DATA_PATH,MODEL_PATH)