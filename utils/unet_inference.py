import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from utils.SatDataset import SatDataset
from utils.unet import UNet
from utils.helpers import *


# Function to do inference on the test dataset
def compute_inference(data_path, model_pth, device, output_dir):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    dataset = SatDataset(data_path, training=False)
    predictions = []
    for id, img in enumerate(dataset):
        img = img.float().to(device)
        img = img.unsqueeze(0)
        mask = model(img)
        mask = mask.squeeze(0).cpu().detach()
        mask = mask.permute(1, 2, 0)
        mask[mask < 0] = 0
        mask[mask > 0] = 1
        predictions.append(mask)
        # Save the mask with an appropriate name
        input_image_name = os.path.basename(dataset.images[id])
        output_name = f"predicted_{os.path.splitext(input_image_name)[0]}.png"
        save_mask(mask, os.path.join(output_dir, output_name))


# Function to do inference on a single image
def single_image_inference(image_pth, model_pth, device, output_dir):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    # We need to apply the same transformation as the training data
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
    mask = model(img)
    mask = mask.squeeze(0).cpu().detach()
    mask = mask.permute(1, 2, 0)
    mask[mask < 0] = 0
    mask[mask > 0] = 1
    save_mask(mask, os.path.join(output_dir, "single_image.png"))


if __name__ == "__main__":
    SINGLE_IMG_PATH = ""
    DATA_PATH = "dataset/test_dataset"
    MODEL_PATH = "models/final_unet.pth"
    OUTPUT_DIR = "dataset/test_dataset/predicted"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    compute_inference(DATA_PATH, MODEL_PATH, device, OUTPUT_DIR)
    # single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device, OUTPUT_DIR)
    print("Done")
