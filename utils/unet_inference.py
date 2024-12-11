"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from utils.SatDataset import SatDataset
from utils.unet import UNet
from utils.helpers import *

import numpy as np
from PIL import Image

def save_mask(mask, path):
    mask = mask.squeeze(0).cpu().detach().numpy()
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]  # Remove the last dimension if it is 1
    Image.fromarray(mask * 255).save(path)

def pred_show_image_grid(data_path, model_pth, device, output_dir):
    model = UNet(in_channels=3, out_channels=1).to(device)
    print("Loading model")
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    print("Model loaded")
    image_dataset = SatDataset(data_path)
    print("Now I am here")
    pred_masks = []

    for idx, (img, orig_mask) in enumerate(image_dataset):
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        pred_masks.append(pred_mask)

        # Save the predicted mask
        save_mask(pred_mask, os.path.join(output_dir, f"predicted_{idx}.png"))

    fig = plt.figure()
    for i in range(1, len(pred_masks) + 1):
        fig.add_subplot(1, len(pred_masks), i)
        plt.imshow(pred_masks[i - 1], cmap="gray")
    plt.show()

def single_image_inference(image_pth, model_pth, device, output_dir):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)

    pred_mask = model(img)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1
    # Save the predicted mask
    print("saving mask")
    save_mask(pred_mask, os.path.join(output_dir, "/single_predicted_2.png"))
    print("mask saved")
    fig = plt.figure()
    for i in range(1, 2):
        fig.add_subplot(1, 1, i)
        plt.imshow(pred_mask, cmap="gray")
    plt.show()

if __name__ == "__main__":
    SINGLE_IMG_PATH = "../dataset/short_testing/images/satImage_075.png"
    DATA_PATH = "../dataset/short_testing"
    MODEL_PATH = "../models/unet.pth"
    OUTPUT_DIR = "../dataset/short_testing/predicted"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, OUTPUT_DIR)
    print("Single Image Inference")
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device, OUTPUT_DIR)
    print("Done")
print("Done")
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from utils.SatDataset import SatDataset
from utils.unet import UNet
from utils.helpers import *


def save_mask(mask, path):
    mask = mask.squeeze(0).cpu().detach().numpy()
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]  # Remove the last dimension if it is 1
    Image.fromarray(mask * 255).save(path)


def batch_inference(data_path, model_pth, device, output_dir, batch_size=4):
    """
    Perform inference on multiple images in batches.

    :param data_path: Path to the dataset folder containing images.
    :param model_pth: Path to the trained model.
    :param device: Device to run the inference ('cuda' or 'cpu').
    :param output_dir: Directory to save predicted masks.
    :param batch_size: Number of images to process at once.
    """
    model = UNet(in_channels=3, out_channels=1).to(device)
    print("Loading model")
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded")

    image_dataset = SatDataset(data_path)
    print("Dataset loaded")

    loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    pred_masks = []

    print("Starting inference")
    for batch_idx, (imgs, orig_masks) in enumerate(loader):
        imgs = imgs.float().to(device)

        with torch.no_grad():
            pred_batch = model(imgs)
            pred_batch = pred_batch.cpu().detach()

        for i, pred_mask in enumerate(pred_batch):
            pred_mask = pred_mask.squeeze(0)
            pred_mask = (pred_mask > 0).float()  # Binarize mask
            pred_masks.append(pred_mask)

            # Save each predicted mask
            save_mask(pred_mask, os.path.join(output_dir, f"predicted_{batch_idx * batch_size + i}.png"))

    # Display results
    print("Inference completed. Visualizing predictions.")
    fig = plt.figure(figsize=(15, 5))
    for i in range(len(pred_masks)):
        fig.add_subplot(1, len(pred_masks), i + 1)
        plt.imshow(pred_masks[i], cmap="gray")
    plt.show()


if __name__ == "__main__":
    DATA_PATH = "../dataset/short_testing"
    MODEL_PATH = "../models/unet.pth"
    OUTPUT_DIR = "../dataset/short_testing/predicted"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    batch_inference(DATA_PATH, MODEL_PATH, device, OUTPUT_DIR, batch_size=4)
    print("Done")
