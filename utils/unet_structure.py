import PIL.Image as Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from utils.unet import UNet

class UNet_Structure:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.cur_epoch = 0
        self.batch_size = 0
        self.mean = 0
        self.std = 0
        # Set the model to UNet
        self.model = UNet().to(torch.device(self.device))
        # declare evaluation metrics array
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy =[]
        self.val_accuracy = []
        self.learning_rate = []
        self.train_positive_f1_score = []
        self.validation_positive_f1_score = []
        self.train_negative_f1_score = []
        self.validation_negative_score = []

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[Image.Image]:
        # Put the model in evaluation mode
        self.model.eval()
        # Transform the input images to tensors, normalize them and stack them
        normalize = self.to_tensor_and_normalize()
        stacked_images = torch.stack([normalize(img) for img in images])
        # Create a DataLoader with all images
        dataloader_images = DataLoader(TensorDataset(stacked_images), batch_size=self.batch_size)
        # Create conversion function from tensor to PIL image
        to_pil_img = transforms.ToPILImage()
        # Making predictions, considering a positive prediction if the value is greater than 0.5
        predictions = []
        for i in dataloader_images:
            i = i[0].to(self.device)
            prediction = self.model(i)
            # Ensure the values are between 0 and 1
            prediction = prediction.clamp(min=0, max=1).round()
            predictions.extend([to_pil_img(pred.cpu()) for pred in prediction])
        return predictions

    # transform the input images to tensors and normalize them
    def to_tensor_and_normalize(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

    def print_training_stats(self, start_epoch: int = 0, end_epoch: int = None):
        """Print the training statistics
        :param start_epoch: The starting epoch to use to print the stats
        :param  end_epoch: The end epoch to use to print the stats
        """
        if end_epoch is not None:
            self._plot_training_curves(end_epoch, start_epoch)
        else:
            self._plot_training_curves(self.cur_epoch)

    # restore the model at a point in time
    def restore_model(self, model_path: str) -> None:
        pass

    # TODO : modify this function
    def _plot_training_curves(self, num_epoch: int, start_epoch: int = 0):
        """Plot train and val accuracy and loss evolution"""

        start_train_idx = start_epoch * self.training_batch_number
        end_train_idx = num_epoch * self.training_batch_number

        n_train = len(self.train_acc_history[start_train_idx:end_train_idx])
        # space train data evenly
        t_train = start_epoch + ((num_epoch - start_epoch) * np.arange(n_train) / n_train)
        # space val data evenly
        t_val = start_epoch + np.arange(1, (num_epoch - start_epoch) + 1)

        fig, ax = plt.subplots(4, 2, figsize=(15, 12))

        # plot accuracy evolution
        ax[0][0].plot(t_train, self.train_acc_history[start_train_idx:end_train_idx], label='Train')
        ax[0][0].plot(t_val, self.val_acc_history[start_epoch:num_epoch], label='Val')
        ax[0][0].legend()
        ax[0][0].set_xlabel('Epoch')
        ax[0][0].set_ylabel('Accuracy')

        # plot loss evolution
        ax[0][1].plot(t_train, self.train_loss_history[start_train_idx:end_train_idx], label='Train')
        ax[0][1].plot(t_val, self.val_loss_history[start_epoch:num_epoch], label='Val')
        ax[0][1].legend()
        ax[0][1].set_xlabel('Epoch')
        ax[0][1].set_ylabel('Loss')

        # Check min value for the two performance metric (training)
        min_neg = np.array(self.train_prf1_n_history[start_train_idx:end_train_idx]).min()
        min_pos = np.array(self.train_prf1_p_history[start_train_idx:end_train_idx]).min()
        min_train = np.min([min_pos, min_neg])
        min_train = 0 if np.isnan(min_train) else min_train

        # plot positive train precision, recall and f1 evolution
        ax[1][0].plot(t_train, self.train_prf1_p_history[start_train_idx:end_train_idx],
                      label=['Precision positive', 'Recall positive', 'f1 positive'])
        ax[1][0].legend()
        ax[1][0].set_xlabel('Epoch')
        ax[1][0].set_ylabel('Training Positive metrics')
        ax[1][0].set_ylim(min_train)

        # plot negative train precision, recall and f1 evolution
        ax[1][1].plot(t_train, self.train_prf1_n_history[start_train_idx:end_train_idx],
                      label=['Precision negative', 'Recall negative', 'f1 negative'])
        ax[1][1].legend()
        ax[1][1].set_xlabel('Epoch')
        ax[1][1].set_ylabel('Training Negative metrics')
        ax[1][1].set_ylim(min_train)

        # Check min value for the two performance metric (validation)
        min_neg = np.array(self.val_prf1_n_history[start_epoch:num_epoch]).min()
        min_pos = np.array(self.val_prf1_p_history[start_epoch:num_epoch]).min()
        min_test = np.min([min_pos, min_neg])
        min_test = 0 if np.isnan(min_test) else min_test

        # plot positive val precision, recall and f1 evolution
        ax[2][0].plot(t_val, self.val_prf1_p_history[start_epoch:num_epoch],
                      label=['Precision positive', 'Recall positive', 'f1 positive'])
        ax[2][0].legend()
        ax[2][0].set_xlabel('Epoch')
        ax[2][0].set_ylabel('Val Positive metrics')
        ax[2][0].set_ylim(min_test)

        # plot negative val precision, recall and f1 evolution
        ax[2][1].plot(t_val, self.val_prf1_n_history[start_epoch:num_epoch],
                      label=['Precision negative', 'Recall negative', 'f1 negative'])
        ax[2][1].legend()
        ax[2][1].set_xlabel('Epoch')
        ax[2][1].set_ylabel('Val Negative metrics')
        ax[2][1].set_ylim(min_test)

        # plot learning rate evolution
        ax[3][0].plot(t_train, self.lr_history[start_train_idx:end_train_idx], label='learning rate')
        ax[3][0].legend()
        ax[3][0].set_xlabel('Epoch')
        ax[3][0].set_ylabel('Learning rate evolution')

        plt.show()