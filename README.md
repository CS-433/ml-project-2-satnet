# Image Processing - Road Segmentation From Satellite Images


This repository contains the code and the report of the project `Road Segmentation From Satellite Images`, which is the project 2 of the 
([CS-433](https://edu.epfl.ch/coursebook/fr/machine-learning-CS-433)) course at the École Polytechnique Fédérale de Lausanne 
([EPFL](https://www.epfl.ch/en/)). This project focuses on binary pixel-wise classification to determine whether each pixel in an image belongs to a road or not, facilitating the segmentation of road areas. The findings of our research and the achieved performance metrics are documented in the PDF file available within this repository.

*Note : The original training and test datasets can be found [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files).
The used datasets and the trained models can be found in the [Google Drive](https://drive.google.com/drive/folders/1iJobZW7g3ZYpGy5liyJC1xlxVRjsFFpq?usp=sharing)


**Authors:** : 
Mahmoud Dokmak, Romain Corbel, Guilhem Destriau

<hr style="clear:both">

## Repository Structure

- `/ml-project-2-satnet`
    - `/dataset`
        - `/augmented_dataset`: contains the augmented dataset with 3000 images that we used to train our final model
        - `/dataset_1000_images`: contains a dataset of 1000 images that we used to train our first cnn models
        - `/TrainingInde`: contains another dataset of 1000 images that we used to train our logistic regression models
        - `/training`: contains the original training dataset
        - `/test_dataset`: contains the original testing dataset reformatted to work with our cnn models, and a folder to store the predictions
        - `/test_set_images`: contains the original testing dataset used to test the logistic regression models
    - `/models`:
      - `/submission.csv`: contains the submission files for the AI Crowd Contest
      - `/final_unet.pth`: savings for the final UNet model trained on the `augmented_dataset` of 3000 images
      - `/metrics_final.json`: contains the performance metrics per epoch of the `final_unet.pth` training (our best model)
      - `/unet.pth`: saving for the UNet model trained on the `dataset_1000_images` of 1000 images
      - `/metrics.json`: contains the performance metrics per epoch of the `unet.pth` training
      - `/cnn_augmented.pth`: savings for CNN model trained on the `augmented_dataset` of 3000 images
      - `/cnn.pth`: savings for CNN model trained on the original `training` dataset of 100 images
      - `/best_model_2d.pkl`: savings for the logistic regression model trained on the original `training` dataset of 100 images on 2 features
      - `/best_model_2d_augm.pkl`: savings for the logistic regression model trained on the `TrainingInde` dataset of 1000 images on 2 features
      - `/best_model_6d.pkl`: savings for the logistic regression model trained on the original `training` dataset of 100 images on 6 features
      - `/best_model_6d_augm.pkl`: savings for the logistic regression model trained on the `TrainingInde` dataset of 1000 images on 6 features
      - `/best_model_8d.pkl`: savings for the logistic regression model trained on the original `training` dataset of 100 images on 8 features
      - `/best_model_8d_augm.pkl`: savings for the logistic regression model trained on the `TrainingInde` dataset of 1000 images on 8 features
    - `/utils`
      - `helpers.py`: implement some useful functions across every model
      - `data_normalization.py`: used to normalize the data
      - `DataAugmentation.py`: used to augment the data
      - `run_data_augmentation.py`: used to simplify the data augmentation
      - `logistic_regression.py`: implement logistic regression model
      - `SatDataset.py`: implement our dataset class
      - `unet.py`: implement U-Net architecture
      - `test_unet.py`: test the U-Net model
      - `unet_trainer.py`: implement the training process for U-Net
      - `unet_inference.py`: implement the inference process for U-Net
      - `cnn_trainer.py`: implement the training process for CNN
      - `cnn_tuning.py`: implement the tuning process for CNN
      - `mask_to_submissions.py`: implement the submission process
    - `ExploratoryDataAnalysis.ipynb`: jupyter notebook summarizing our data analysis
    - `random.ipynb`: jupyter notebook summarizing our random model
    - `LogisticRegressionfct.ipynb`: jupyter notebook summarizing our logistic regression models
    - `run.ipynb`: jupyter notebook to run to train, test and submit the U-Net model. It is  the one to run to be able to reproduce our `submission.csv` file
    - `report.pdf`: the report of our project

## Usage
<hr style="clear:both">

To generate our submission file, you only need the download from this [Google Drive](https://drive.google.com/drive/folders/1iJobZW7g3ZYpGy5liyJC1xlxVRjsFFpq?usp=sharing) the `final_unet.pth` model (which corresponds to our best model, the U-Net model trained on the `augmented_dataset` of 3000 images) and put it in `models` (You can also download the whole `models` folder and replace the current empty one). Then, you can run the `run.ipynb` notebook to train, test and create the `submission.csv` file.
As default, we commented the training part, so by running all, only the inference and the creation of the csv is performed. We recommend to run the notebook on Google Colab as it is the environment we used to train and test our models. Furthermore, the notebook is already set up to work on Google Colab.

By downloading the whole models and dataset folders and replacing the currents ones, you can reproduce all our results (train, preform inference etc).
If you want to train the U-Net model, you should additionaly uncomment the training line in the `run.ipynb` notebook.