import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class SatDataset(Dataset):
    def __init__(self, root_path, training=True):
        self.root_path = root_path
        self.training = training
        # We sort the images to make sure they are aligned
        self.images = sorted([root_path + "/images/" + i for i in os.listdir(root_path + "/images/")])
        # We only have groundtruth if we are training, not for testing
        if training:
            self.ground = sorted([root_path + "/groundtruth/" + i for i in os.listdir(root_path + "/groundtruth/")])
        mean = [0.3305, 0.3261, 0.2917]
        std = [0.1894, 0.1836, 0.1829]
        # Transform for images: Convert to tensor and normalize
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Transform for masks: Convert to tensor (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # Open images in RGB mode and apply the transformation
        img = Image.open(self.images[index]).convert("RGB")
        img = self.img_transform(img)
        if self.training:
            # Open masks in L mode and apply the transformation
            mask = Image.open(self.ground[index]).convert("L")
            mask = self.mask_transform(mask)
            return img, mask
        else:
            return img

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)