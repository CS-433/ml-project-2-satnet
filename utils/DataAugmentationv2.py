"""
Helper functions used to increase the samples/images in the training data set.
"""

from os import listdir, path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


def DataAugmentation(nb: int, training_folder: str, augmented_image_folder: str, augmented_groundtruth_folder: str,
                    split_images: bool = False):
    """
    Function that augments the dataset with some transformation applied to images, and save them to disk

    :param nb: The number of time to pass the whole dataset into transformation process
               (i.e. nb=10 goes from 100 to 1100 images, by adding 1000 new images to the 100 existing ones)
    :param training_folder: the folder containing the images and groundtruth
    :param augmented_image_folder: the folder where to store the augmented images
    :param augmented_groundtruth_folder: the folder where to store the augmented groundtruth
    :param add_base_images: whether to add the base images to the augmented dataset or not
    :param split_images: whether the images are split or not
    """



    train_set = DatasetAugmentation(training_path=training_folder, split_images=split_images, perform_transformations=False)
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=False)
    nb_images = len(train_dataloader)
    to_pil = v2.ToPILImage()
    
    #initialisation of the dataset

    for idx, (image, groundtruth) in enumerate(train_dataloader):
        # get transformed image and groundtruth from dataloader
        image = to_pil(image.squeeze(0))
        groundtruth = to_pil(groundtruth.squeeze(0))

        # save image to disk
        image.save(path.join(training_folder, augmented_image_folder, f'satImage_{idx + 1:06d}.png'))
        groundtruth.save(path.join(training_folder, augmented_groundtruth_folder, f'satImage_{idx + 1:06d}.png'))

    transform_train_set = DatasetAugmentation(training_path=training_folder, split_images=split_images, perform_transformations=True)
    transform_train_dataloader = DataLoader(transform_train_set, batch_size=1, shuffle=False)
    for i in range(nb):
        for idx, (image, groundtruth) in enumerate(transform_train_dataloader):
            image = to_pil(image.squeeze(0))
            groundtruth = to_pil(groundtruth.squeeze(0))

            # compute the image index
            image_idx =  (i+1) * nb_images + idx + 1
            # save image to disk
            image.save(path.join(training_folder, augmented_image_folder, f'satImage_{image_idx:06d}.png'))

            # save gt to disk
            groundtruth.save(path.join(training_folder, augmented_groundtruth_folder, f'satImage_{image_idx:06d}.png'))



class DatasetAugmentation(Dataset):

    def __init__(self, training_path, split_images=False, perform_transformations=True):
        self.training_path = training_path
        self.split = split_images
        self.transformation = perform_transformations
        if split_images:
            self.images = sorted(listdir(path.join(training_path, 'split_images')))
            self.groundtruth = sorted(listdir(path.join(training_path, 'split_groundtruth')))
        else:
            self.images = sorted(listdir(path.join(training_path, 'images')))
            self.groundtruth = sorted(listdir(path.join(training_path, 'groundtruth')))
        
        #define all the possible transformations
        self.CropResized = v2.RandomResizedCrop(size=400, antialias=True)
        self.FlipHorizotale = v2.RandomHorizontalFlip(p=0.5)
        self.FlipVertical = v2.RandomVerticalFlip(p=0.5)
        self.Rotation = v2.RandomApply([v2.RandomRotation(360)], p=0.5)
        self.Blur = v2.RandomApply([v2.GaussianBlur(5)], p=0.5)
        self.Brightness = v2.RandomApply([v2.ColorJitter(brightness=(0.5, 1.5))], p=0.5)
        self.Contrast = v2.RandomApply([v2.ColorJitter(contrast=(0.5, 1.5))], p=0.25)
        self.Saturation = v2.RandomApply([v2.ColorJitter(saturation=(0.5, 1.5))], p=0.25)

    def __getitem__(self,image_idx):

        if self.split:
            image_path = path.join(self.training_path,'split_images', self.images[image_idx])
            groundtruth_path = path.join(self.training_path,'split_groundtruth', self.groundtruth[image_idx])
        else:
            image_path = path.join(self.training_path,'images', self.images[image_idx])
            groundtruth_path = path.join(self.training_path,'groundtruth', self.groundtruth[image_idx])
        
        image = read_image(image_path, ImageReadMode.RGB)
        groundtruth = read_image(groundtruth_path, ImageReadMode.RGB)

        regroupment = torch.stack([image, groundtruth], dim=0)

        if self.transformation:
            if not self.split:
                regroupment = self.CropResized(regroupment)
            regroupment = self.FlipHorizotale(regroupment)
            regroupment = self.FlipVertical(regroupment)

            #separate again Image and Groundtruth in order to modify only the Image
            image, groundtruth = regroupment[0], regroupment[1]

            #perform transformation for the Image 
            image = self.Blur(image)
            image = self.Brightness(image)
            image = self.Contrast(image)
            image = self.Saturation(image)

        # Convert ground truth back to grayscale
        groundtruth = v2.Grayscale()(groundtruth)

        return image, groundtruth
    
    def __len__(self):
        return len(self.images)



