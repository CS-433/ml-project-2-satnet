import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class SatDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        # We sort the images and masks to make sure they are aligned
        self.images = sorted([root_path + "/images/" + i for i in os.listdir(root_path + "/images/")])
        self.ground = sorted([root_path + "/groundtruth/" + i for i in os.listdir(root_path + "/groundtruth/")])
        print("We accessed the images and masks")
        # transform to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        # Open images and masks in RGB and L mode respectively and apply the transformation
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.ground[index]).convert("L")
        return self.transform(img), self.transform(mask)

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)
