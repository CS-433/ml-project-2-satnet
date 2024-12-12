import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from utils.SatDataset import SatDataset
from utils.unet import UNet
from utils.helpers import array_to_submission

def pred_show_image_grid(data_path, model_pth, device, output_file):
    model = UNet(in_channels=3, out_channels=1).to(device)
    print("Loading model")
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    print("Model loaded")
    image_dataset = SatDataset(data_path)
    print("Now I am here")
    predictions = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
        pred_mask = (pred_mask > 0).astype(np.uint8)  # Convert to binary mask

        predictions.extend(pred_mask.flatten())

    sqrt_n_patches = int(np.sqrt(len(predictions)))
    patch_size = 16  # Assuming patch size is 16, adjust if different
    array_to_submission(output_file, predictions, sqrt_n_patches, patch_size)
    print(f"Predictions saved to {output_file}")

def single_image_inference(image_pth, model_pth, device, output_file):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)

    pred_mask = model(img)

    pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
    pred_mask = (pred_mask > 0).astype(np.uint8)  # Convert to binary mask

    predictions = pred_mask.flatten()
    sqrt_n_patches = int(np.sqrt(len(predictions)))
    patch_size = 16  # Assuming patch size is 16, adjust if different
    array_to_submission(output_file, predictions, sqrt_n_patches, patch_size)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    SINGLE_IMG_PATH = "../dataset/short_testing/images/SatImage_075.png"
    DATA_PATH = "../dataset/short_testing"
    MODEL_PATH = "../models/unet.pth"
    OUTPUT_FILE = "../unet_predictions.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, OUTPUT_FILE)
    print("Single Image Inference")
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device, OUTPUT_FILE)
    print("Done")