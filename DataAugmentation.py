"""
Helper functions used to increase the samples/images in the training data set.
"""

from os import listdir, path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2


def __initiate_augmented_folder(training_folder: str, augmented_image_folder: str,
                                augmented_groundtruth_folder: str, split_images: bool):
    """
    Function that initiate the folder where to save the augmented images and store the original images

    :param training_folder: the folder containing the images and ground truth
    :param augmented_image_folder: the folder where to store the augmented images
    :param augmented_groundtruth_folder: the folder where to store the augmented groundtruth
    :param split_images: whether the images are split or not
                         (if they are already split, we won't perform random cropping)

    """
    train_set = RoadDatasetAugmentation(root=training_folder, split_images=split_images, perform_transformations=False)
    train_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)

    to_pil = v2.ToPILImage()

    for idx, (img, gt) in enumerate(train_dataloader):
        # get transformed image from dataloader
        img = to_pil(img.squeeze(0))
        # get transformed ground truth from dataloader
        gt = to_pil(gt.squeeze(0))

        # save image to disk
        img.save(path.join(training_folder, augmented_image_folder, f'satImage_{idx + 1:06d}.png'))
        # save gt to disk
        gt.save(path.join(training_folder, augmented_groundtruth_folder, f'satImage_{idx + 1:06d}.png'))


def data_augmentation(nb: int, training_folder: str, augmented_image_folder: str, augmented_groundtruth_folder: str,
                      add_base_images: bool = True, split_images: bool = False):
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

    train_set = RoadDatasetAugmentation(root=training_folder, split_images=split_images, perform_transformations=True)
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=False)
    nb_images = len(train_dataloader)

    start_output_idx = 1
    if add_base_images:
        # add base images to augmented dataset
        __initiate_augmented_folder(training_folder, augmented_image_folder, augmented_groundtruth_folder, split_images)
        start_output_idx += nb_images

    to_pil = v2.ToPILImage()
    print(nb_images)
    print(start_output_idx)
    for i in range(nb):
        for idx, (img, gt) in enumerate(train_dataloader):
            # get transformed image from dataloader
            img = to_pil(img.squeeze(0))
            # get transformed ground truth from dataloader
            gt = to_pil(gt.squeeze(0))

            # compute the image index
            img_idx = start_output_idx + i * nb_images + idx
            # save image to disk
            img.save(path.join(training_folder, augmented_image_folder, f'satImage_{img_idx:06d}.png'))

            # save gt to disk
            gt.save(
                path.join(training_folder, augmented_groundtruth_folder, f'satImage_{img_idx:06d}.png'))


class RoadDatasetAugmentation(Dataset):
    """
    Custom Road Dataset for performing data augmentation.
    """

    def __init__(self, root, split_images=False, perform_transformations=True):
        self.root = root
        self.split_images = split_images
        self.perform_transformations = perform_transformations

        image_dir = 'images_split' if split_images else 'images'
        gt_dir = 'groundtruth_split' if split_images else 'groundtruth'

        self.imgs = sorted(listdir(path.join(root, image_dir)))
        self.gt = sorted(listdir(path.join(root, gt_dir)))

        # Base transformations
        self.transform_base = v2.Compose([
            v2.Identity() if split_images else v2.RandomResizedCrop(size=400, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])

        # Transformations applied with specific probabilities
        self.transform_rotation = v2.RandomApply([v2.RandomRotation(360)], p=0.5)
        self.transform_blur = v2.RandomApply([v2.GaussianBlur(5)], p=0.5)
        self.transform_brightness = v2.RandomApply([v2.ColorJitter(brightness=(0.5, 1.5))], p=0.5)
        self.transform_contrast = v2.RandomApply([v2.ColorJitter(contrast=(0.5, 1.5))], p=0.25)
        self.transform_saturation = v2.RandomApply([v2.ColorJitter(saturation=(0.5, 1.5))], p=0.25)

    def __getitem__(self, idx):
        # Load image and ground truth
        img_path = path.join(
            self.root,
            'images_split' if self.split_images else 'images',
            self.imgs[idx]
        )
        gt_path = path.join(
            self.root,
            'groundtruth_split' if self.split_images else 'groundtruth',
            self.gt[idx]
        )

        img = read_image(img_path, ImageReadMode.RGB)
        gt = read_image(gt_path, ImageReadMode.RGB)

        # Stack the image and ground truth for consistent transformations
        both = torch.stack([img, gt], dim=0)

        if self.perform_transformations:
            # Apply base transformations
            both = self.transform_base(both)
            both = self.transform_rotation(both)

        img, gt = both[0], both[1]

        # Additional transformations for the image only
        if self.perform_transformations:
            img = self.transform_blur(img)
            img = self.transform_brightness(img)
            img = self.transform_contrast(img)
            img = self.transform_saturation(img)

        # Convert ground truth back to grayscale
        gt = v2.Grayscale()(gt)

        return img, gt

    def __len__(self):
        return len(self.imgs)
    


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
        return len(self.imgages)



