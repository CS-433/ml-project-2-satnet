import matplotlib.image as mpimg
import numpy as np
from PIL import Image
# Helper functions


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


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

def value_to_class(v, foreground_threshold = 0.25):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
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
            
            
"""
Helper functions to perform cross validation on logistic regression
"""

import warnings

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


'''
def lin_reg_best_params(penalties, class_weights, x, y):
    """
    Perform cross validation to find the best parameters for a logistic regression model

    :param penalties: list of penalties to try
    :param class_weights: list of class weights to try
    :param x: the features
    :param y: the labels
    :return: the best parameters and the best f1 score
    """
    # disable warnings for readability
    warnings.filterwarnings('ignore')

    best_score = -1
    best_params = None

    for penalty in penalties:
        if penalty is None:
            solvers = ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            cs = [1]
            l1_ratios = [0]
        elif penalty == 'l1':
            solvers = ['liblinear', 'saga']
            cs = np.logspace(-8, 10, 2)
            l1_ratios = [1]
        elif penalty == 'l2':
            solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            cs = np.logspace(-8, 10, 2)
            l1_ratios = [0]
        else:
            solvers = ['saga']
            cs = np.logspace(-8, 10, 2)
            l1_ratios = np.linspace(0, 1, 10)

        for solver in solvers:
            for C in cs:
                for class_weight in class_weights:
                    for l1_ratio in l1_ratios:
                        # evaluate the model
                        logreg = linear_model.LogisticRegression(C=C,
                                                                 penalty=penalty,
                                                                 class_weight=class_weight,
                                                                 solver=solver,
                                                                 l1_ratio=l1_ratio,
                                                                 max_iter=1000)
                        f1_score = cross_val_score(logreg, x, y, cv=10, scoring="f1").mean()
                        # update best score and parameters if necessary
                        if f1_score > best_score:
                            best_score = f1_score
                            best_params = {'penalty': penalty,
                                           'C': C,
                                           'solver': solver,
                                           'class_weight': class_weight,
                                           'l1_ratio': l1_ratio}
    warnings.filterwarnings("default")
    return best_params, best_score
'''

