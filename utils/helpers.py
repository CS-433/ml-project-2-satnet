"""
Helper functions to load and display images
"""

import os
import torch
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from utils.cnn import SatelliteRoadCNN
from utils.SatDataset import SatDataset
from torch.utils.data import DataLoader

def value_to_class(v, foreground_threshold = 0.25):
    """
    Classifies the input value `v` based on a foreground threshold.

    Parameters:
    v (array-like): A numerical array or list where the sum is evaluated.
    foreground_threshold (float, optional): The threshold above which the input is classified as '1'. 
                                            Default is 0.25.

    Returns:
    int: Returns 1 if the sum of the values in `v` exceeds the threshold, otherwise returns 0.
    """
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def load_image(infilename):
    """
    Loads an image from a specified file path.

    Parameters:
    infilename (str): The file path of the image to be loaded.

    Returns:
    numpy.ndarray: The image data as a NumPy array, with pixel values.
    """
    data = mpimg.imread(infilename)
    return data

def load_data(folder_path, is_test=False):
    """
    Loads images and groundtruth data from the specified folder path. Can handle both training and testing data.

    Parameters:
    folder_path (str): The root folder path containing the data.
    is_test (bool, optional): If True, loads test data (images only). 
                               If False, loads training data (images and groundtruth). Default is False.

    Returns:
    tuple: 
        - A list of loaded images as NumPy arrays.
        - The number of files (n_files).
        - A list of file names (for training data).
        - A list of groundtruth images (for training data), or None (for test data).
    """
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
    """
    Converts a floating point image to an unsigned 8-bit integer image.

    This function normalizes the floating point image to the range [0, 255]
    and then converts it to uint8 format.

    Parameters:
    img (numpy.ndarray): A floating-point image, typically with values in [0, 1] or any floating point range.

    Returns:
    numpy.ndarray: The image converted to uint8 format with values in the range [0, 255].
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """
    Concatenates an image (`img`) and a ground truth image (`gt_img`) side by side.
    
    If the ground truth image is grayscale, it is converted to a 3-channel image 
    by replicating the grayscale values across the 3 channels. The final concatenated
    image will be in RGB format.

    Parameters:
    img (numpy.ndarray): The input image to be concatenated, which can be a 3-channel (RGB) image.
    gt_img (numpy.ndarray): The ground truth image, which can be either grayscale or 3-channel (RGB).

    Returns:
    numpy.ndarray: A concatenated image where `img` and `gt_img` are placed side by side.
    """
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
    """
    Crops an image into smaller patches of size (w, h).

    The function divides the input image into patches of the specified width (`w`) 
    and height (`h`), and returns a list of the cropped patches.

    Parameters:
    im (numpy.ndarray): The input image to be cropped, which can be either 2D (grayscale) or 3D (RGB).
    w (int): The width of each patch.
    h (int): The height of each patch.

    Returns:
    list: A list of image patches (numpy arrays) of size (w, h) (or (w, h, 3) for RGB images).
    """
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
    """
    Standardizes the input training and testing datasets by scaling them to have zero mean and unit variance.

    The function uses the StandardScaler from scikit-learn to standardize both the training and testing data. 
    It first fits the scaler to the training data and then transforms both the training and testing data.

    Parameters:
    X_train (numpy.ndarray): The training dataset, where rows are samples and columns are features.
    X_test (numpy.ndarray): The testing dataset, where rows are samples and columns are features.

    Returns:
    tuple: 
        - X_train_scaled (numpy.ndarray): The standardized training dataset.
        - X_test_scaled (numpy.ndarray): The standardized testing dataset.
        - scaler (StandardScaler): The fitted StandardScaler object, which can be used to transform other datasets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled , scaler

def extract_patches(patch_size,imgs,n):
    """
    Extracts patches of a specified size from a list of images.

    This function divides each image in the `imgs` list into smaller patches of size 
    `(patch_size, patch_size)` using the `img_crop` function, and then flattens the list 
    of patches into a 1D array.

    Parameters:
    patch_size (int): The width and height of the square patches to be extracted.
    imgs (list of numpy.ndarray): A list of images to be cropped, each of shape (height, width, channels).
    n (int): The number of images in the list `imgs` to process.

    Returns:
    numpy.ndarray: A 1D array of image patches, where each patch is of size (patch_size, patch_size) 
                    or (patch_size, patch_size, channels) depending on the image format.
    """
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    img_patches = np.array(img_patches)
    print(f"Shape of unflattened patches : {img_patches.shape}")
    img_patches = np.asarray(
        [
            img_patches[i][j]
            for i in range(len(img_patches))
            for j in range(len(img_patches[i]))
        ]
    )
    print(f"Shape of flattened patches : {img_patches.shape}\n")
    return img_patches


def extract_features(img):
    """
    Extracts basic statistical features (mean and variance) from an image.

    This function computes the mean and variance of the pixel values for each color channel
    (or for the entire image if grayscale), then concatenates these values into a single feature vector.

    Parameters:
    img (numpy.ndarray): The input image, which can be either grayscale or RGB (height x width x channels).

    Returns:
    numpy.ndarray: A feature vector containing the mean and variance for each color channel.
                    The vector has a length of `2 * num_channels` (mean + variance for each channel).
    """
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_img_features(filename, patch_size = 16):
    """
    Extracts feature vectors from an image by dividing it into patches and computing statistical features for each patch.

    This function loads an image from the provided filename, divides it into square patches of the specified size, 
    and extracts basic features (mean and variance of pixel values) from each patch. The extracted features are 
    returned as a 2D array where each row corresponds to a patch.

    Parameters:
    filename (str): The path to the image file from which features are to be extracted.
    patch_size (int, optional): The size of the square patches (height and width). Default is 16.

    Returns:
    numpy.ndarray: A 2D array where each row is a feature vector extracted from a patch. 
                    The number of rows corresponds to the number of patches, and each feature vector 
                    contains the mean and variance of pixel values for each color channel.
    """
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features(img_patches[i]) for i in range(len(img_patches))]
    )
    return X


def extract_features_2d(img):
    """
    Extracts basic statistical features (mean and variance) from a 2D image.

    This function computes the mean and variance of the pixel values across the entire 2D image.

    Parameters:
    img (numpy.ndarray): The input 2D image (height x width). It is assumed to be in grayscale format.

    Returns:
    numpy.ndarray: A feature vector containing the mean and variance of the pixel values in the image.
                    The vector has length 2, with the first element being the mean and the second being the variance.
    """
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


# Extract features for a given image
def extract_img_features_2d(filename, patch_size = 16):
    """
    Extracts feature vectors from a 2D image by dividing it into patches and computing statistical features for each patch.

    This function loads a 2D image (grayscale), divides it into square patches of the specified size, 
    and extracts basic statistical features (mean and variance of pixel values) for each patch. 
    The extracted features are returned as a 2D array where each row corresponds to a patch.

    Parameters:
    filename (str): The path to the image file from which features are to be extracted.
    patch_size (int, optional): The size of the square patches (height and width). Default is 16.

    Returns:
    numpy.ndarray: A 2D array where each row is a feature vector extracted from a patch. 
                    The number of rows corresponds to the number of patches, and each feature vector 
                    contains the mean and variance of pixel values for the entire patch.
    """
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    return X


    
def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Reconstructs an image from a list of patch labels.

    This function takes a list of labels (one per patch), and reconstructs 
    the full image by assigning the corresponding label to the correct 
    position of each patch in the image.

    Parameters:
    imgwidth (int): The width of the final image.
    imgheight (int): The height of the final image.
    w (int): The width of each patch.
    h (int): The height of each patch.
    labels (list or numpy.ndarray): A list or array of labels, one per patch, 
                                    that will be used to fill the image.

    Returns:
    numpy.ndarray: The reconstructed image where each patch is filled with its corresponding label.
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im


def make_img_overlay(img, predicted_img):
    """
    Superimposes a color mask (from predicted labels) onto the original image.

    This function creates a color mask based on the predicted labels (binary or multi-class) 
    and overlays it on top of the original image, with a specified transparency. The original 
    image is displayed with a slight transparency, and the mask is blended with it for visualization.

    Parameters:
    img (numpy.ndarray): The original image to overlay the mask on. It should be in the range [0, 1] (float).
    predicted_img (numpy.ndarray): A binary or multi-class image where each pixel is a predicted label. 
                                    It is used to create the color mask. Values are expected to be in [0, 1].

    Returns:
    PIL.Image.Image: The resulting image with the mask overlayed on the original image, in RGBA format (including transparency).
    """
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
    Generates a CSV file for submission with image patch predictions.

    This function takes the predicted values of image patches, along with the dimensions 
    of the patches and the number of patches per image, and writes them to a CSV file 
    in the required submission format. Each row in the CSV contains the ID of the image 
    (with the patch coordinates) and the predicted value for that patch.

    Parameters:
    submission_filename (str): The path to the output CSV file where the predictions will be written.
    array (numpy.ndarray): A 1D array containing the predicted values for all patches.
                           The array should be of length equal to `sqrt_n_patches ** 2` * number of images.
    sqrt_n_patches (int): The square root of the number of patches along one dimension of the image.
                           Used to compute the grid layout of patches (e.g., for 16x16 patches, `sqrt_n_patches` is 4).
    patch_size (int): The size of each patch (both height and width). Used to calculate the pixel coordinates of patches.

    Returns:
    None: Writes the predictions directly to the specified CSV file.
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for index, pixel in enumerate(array):
            img_number = 1 + index // (sqrt_n_patches ** 2)
            j = patch_size * ((index // sqrt_n_patches) % sqrt_n_patches)
            i = patch_size * (index % sqrt_n_patches)
            f.writelines(f'{img_number:03d}_{j}_{i},{pixel}\n')

def save_mask(mask, path):
    """
    Saves a binary mask as an image file.

    This function takes a tensor or array representing a binary mask, converts it to a 
    NumPy array, thresholds it to ensure it's binary (values of 0 or 1), and then saves 
    it as an image file. The mask will be saved with pixel values in the range [0, 255] 
    for visualization.

    Parameters:
    mask (torch.Tensor or numpy.ndarray): The input mask, typically a tensor that 
                                          needs to be squeezed and converted to a binary format.
    path (str): The file path where the mask image will be saved. The file extension 
                should be supported by PIL (e.g., `.png`, `.jpg`, etc.).

    Returns:
    None: Saves the image file at the specified location.
    """
    mask = mask.squeeze(0).cpu().detach().numpy()
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]  # Remove the last dimension if it is 1
    Image.fromarray(mask * 255).save(path)

def calculate_metrics(y_pred, y_true):
    """
    Calculates accuracy and F1-score for binary classification.

    This function computes two common metrics used for evaluating binary classification performance:
    - Accuracy: The proportion of correct predictions (both true positives and true negatives) out of all predictions.
    - F1-score: The harmonic mean of precision and recall, providing a balance between the two.

    The function assumes that `y_pred` and `y_true` are either PyTorch tensors or NumPy arrays and that
    they represent binary classification results.

    Parameters:
    y_pred (torch.Tensor or numpy.ndarray): The predicted binary labels (values between 0 and 1).
    y_true (torch.Tensor or numpy.ndarray): The true binary labels (values between 0 and 1).

    Returns:
    tuple: A tuple containing the accuracy and F1-score of the prediction.
    """
    y_pred = (y_pred > 0).float()
    y_true = (y_true > 0).float()
    accuracy = accuracy_score(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())
    f1 = f1_score(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())
    return accuracy, f1

def find_best_image(device, root,model_pth,treshhold):
    """
    Finds the image with the best F1-score by comparing the predicted mask with the ground truth.

    This function loads a pre-trained segmentation model, evaluates it on a dataset of satellite images,
    and computes the F1-score for each image. It returns the image with the highest F1-score, along with 
    the predicted mask and the ground truth mask for that image.

    Parameters:
    device (torch.device): The device (CPU or GPU) where the model and data should be loaded.
    root (str): The root directory where the dataset of satellite images is stored.
    model_pth (str): The path to the saved model checkpoint (.pth file).
    threshold (float): The threshold value used to binarize the predicted mask.

    Returns:
    img_good (torch.Tensor): The image with the highest F1-score.
    mask_good (torch.Tensor): The predicted mask for the image with the highest F1-score.
    grt_good (torch.Tensor): The ground truth mask for the image with the highest F1-score.
    f1_max (float): The highest F1-score achieved.
    """
    model = SatelliteRoadCNN().to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = SatDataset(root)
    train_dataloader = DataLoader(dataset=image_dataset,
                                  batch_size=1,
                                  shuffle=False)
    f1_max = 0
    
    for idx, img_mask in enumerate(train_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            mask = mask.squeeze(0).squeeze(0)
            mask=(mask>=0.5).int()
            grt = mask.cpu().numpy().flatten()
            pred_mask = model(img)
            pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().detach()
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask >= treshhold).int().cpu()
            y_pred = pred_mask.numpy().flatten()
            f1 = f1_score(grt,y_pred)
            if(f1>f1_max):
                f1_max = f1
                img_good = img.squeeze(0).squeeze(0).cpu()
                mask_good = pred_mask
                grt_good = mask.squeeze(0).squeeze(0).cpu()
    return img_good, mask_good,grt_good, f1_max

def metrics_mean_std(device, root,model_pth,treshhold):
    """
    Calculates the mean and standard deviation of accuracy and F1-score for a segmentation model.

    This function evaluates the performance of a pre-trained segmentation model on a dataset of satellite images.
    For each image, it computes both the accuracy and the F1-score by comparing the predicted mask with the ground truth.
    It then returns the mean and standard deviation of these metrics across the whole dataset.

    Parameters:
    device (torch.device): The device (CPU or GPU) where the model and data should be loaded.
    root (str): The root directory where the dataset of satellite images is stored.
    model_pth (str): The path to the saved model checkpoint (.pth file).
    threshold (float): The threshold value used to binarize the predicted mask (usually between 0 and 1).

    Returns:
    tuple: A tuple containing the following values:
        - mean_accuracy (float): The mean accuracy across all images.
        - std_accuracy (float): The standard deviation of accuracy across all images.
        - mean_f1 (float): The mean F1-score across all images.
        - std_f1 (float): The standard deviation of F1-scores across all images.
    """
    model = SatelliteRoadCNN().to(device)
    print("Loading model")
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = SatDataset(root)
    train_dataloader = DataLoader(dataset=image_dataset,
                                  batch_size=1,
                                  shuffle=False)
    
    accuracies = []
    f1scores = []
    for idx, img_mask in enumerate(train_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            mask = mask.squeeze(0).squeeze(0)
            mask=(mask>=0.5).int()
            grt = mask.cpu().numpy().flatten()
            pred_mask = model(img)
            pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().detach()
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask >= treshhold).int().cpu()
            y_pred = pred_mask.numpy().flatten()
            f1 = f1_score(grt,y_pred)
            acc = accuracy_score(grt,y_pred)
            accuracies.append(acc)
            f1scores.append(f1)
    return np.mean(accuracies),np.std(accuracies), np.mean(f1scores), np.std(f1scores)