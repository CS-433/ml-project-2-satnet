# Compute the mean and standard deviation of the augmented training dataset images.
# We'll apply the same normalization (with the mean and variance of the training set)
# to the validation and test sets.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils.SatDataset import SatDataset

DATA_PATH = "dataset/augmented_dataset/training"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# Use SatDataset with no normalization for this task.
dataset = SatDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Initialize sums and pixel counts
sum_rgb = torch.zeros(3, device=device)
sum_rgb_squared = torch.zeros(3, device=device)
pixels = 0

# Compute for every image in the dataset
for img, grt in dataloader:
    img = img.to(device)
    pixels += img.size(0) * img.size(2) * img.size(3)
    sum_rgb += img.sum(dim=[0, 2, 3])
    sum_rgb_squared += (img**2).sum(dim=[0, 2, 3])
# Compute the mean and standard deviation
mean = sum_rgb / pixels
variance = (sum_rgb_squared / pixels) - (mean**2)
std = torch.sqrt(variance)

# Print the results, when I'll have modifier the SatDataset class,
# the mean will be [0, 0, 0} and the std will be [1, 1, 1].
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
