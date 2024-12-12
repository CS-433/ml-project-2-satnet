import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from SatDataset import SatDataset  # Import your dataset class

# Define dataset
DATA_PATH = "../dataset/mahmoud_training"
dataset = SatDataset(DATA_PATH)

# DataLoader for efficient computation
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Initialize sums and pixel counts
sum_channels = torch.zeros(3)
sum_squared_channels = torch.zeros(3)
num_pixels = 0

# Iterate through the dataset
for images, _ in dataloader:  # Assuming your dataset returns (image, mask)
    images = images  # Shape: (batch_size, 3, height, width)
    num_pixels += images.size(0) * images.size(2) * images.size(3)  # Total number of pixels per channel
    sum_channels += images.sum(dim=[0, 2, 3])  # Sum of all pixel values per channel
    sum_squared_channels += (images ** 2).sum(dim=[0, 2, 3])  # Sum of squares of all pixel values per channel

# Compute mean and std
mean = sum_channels / num_pixels
variance = (sum_squared_channels / num_pixels) - (mean ** 2)
std = torch.sqrt(variance)

# Print the results
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
