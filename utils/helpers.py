import os
import torch
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Helper functions
def value_to_class(v, foreground_threshold = 0.25):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def load_data(folder_path, is_test=False):
    if is_test:
        # For test data
        folder_test = os.listdir(folder_path)
        n_files = len(folder_test)
        images = [
            load_image(os.path.join(folder_path, folder_test[i], f"{folder_test[i]}.png"))
            for i in range(n_files)
        ]
        return images, n_files, folder_test
    else:
        # For training data
        image_dir = os.path.join(folder_path, "images/")
        gt_dir = os.path.join(folder_path, "groundtruth/")
        file_names = os.listdir(image_dir)
        n_files = len(file_names)

        images = [load_image(os.path.join(image_dir, file_names[i])) for i in range(n_files)]
        groundtruth = [load_image(os.path.join(gt_dir, file_names[i])) for i in range(n_files)]

        return images, groundtruth, n_files, file_names


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def standardization(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled , scaler

def extract_patches(patch_size,imgs,n):
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]

    # Convert to numpy arrays
    img_patches = np.array(img_patches)

    print(f"Shape of unflattened patches : {img_patches.shape}")

    # Linearize list of patches
    img_patches = np.asarray(
        [
            img_patches[i][j]
            for i in range(len(img_patches))
            for j in range(len(img_patches[i]))
        ]
    )
    print(f"Shape of flattened patches : {img_patches.shape}\n")
    return img_patches
# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat


# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


# Extract features for a given image
def extract_img_features_2d(filename, patch_size = 16):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    return X

def extract_img_features(filename, patch_size = 16):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features(img_patches[i]) for i in range(len(img_patches))]
    )
    return X
    
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def array_to_submission(submission_filename, array, sqrt_n_patches, patch_size):
    """
    Generates a csv file of predictions from the given array of patches

    :param submission_filename: the filename of the csv file
    :param array: the array of patches
    :param sqrt_n_patches: the square root of the number of patches per image
    :param patch_size: the width and height in pixels of each patch
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for index, pixel in enumerate(array):
            img_number = 1 + index // (sqrt_n_patches ** 2)
            j = patch_size * ((index // sqrt_n_patches) % sqrt_n_patches)
            i = patch_size * (index % sqrt_n_patches)
            f.writelines(f'{img_number:03d}_{j}_{i},{pixel}\n')

def save_mask(mask, path):
    mask = mask.squeeze(0).cpu().detach().numpy()
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]  # Remove the last dimension if it is 1
    Image.fromarray(mask * 255).save(path)

def calculate_metrics(y_pred, y_true):
    y_pred = (y_pred > 0).float()
    y_true = (y_true > 0).float()
    accuracy = accuracy_score(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())
    f1 = f1_score(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())
    return accuracy, f1