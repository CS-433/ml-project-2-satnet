from DataAugmentationv2 import *
import os
from os import path
import random
import shutil
from datetime import datetime
root_training_folder='dataset/training'
augmented_image_folder='images_augmented'
augmented_groundtruth_folder='groundtruth_augmented'
augmentation_factor=19
patch_size=400

image_augmented_folder = path.join(root_training_folder, augmented_image_folder)
groundtruth_augmented_folder = path.join(root_training_folder, augmented_groundtruth_folder)
split_images_folder = path.join(root_training_folder, 'images_split')
split_gt_folder = path.join(root_training_folder, 'groundtruth_split')

# Cleanup training folder
shutil.rmtree(image_augmented_folder, ignore_errors=True)
shutil.rmtree(groundtruth_augmented_folder, ignore_errors=True)
os.makedirs(image_augmented_folder)
os.makedirs(groundtruth_augmented_folder)

shutil.rmtree(split_images_folder, ignore_errors=True)
shutil.rmtree(split_gt_folder, ignore_errors=True)


# perform data augmentation
DataAugmentation(nb=augmentation_factor, training_folder=root_training_folder,
                    augmented_image_folder=augmented_image_folder,
                    augmented_groundtruth_folder=augmented_groundtruth_folder, split_images=False)

print('Data augmentation done')