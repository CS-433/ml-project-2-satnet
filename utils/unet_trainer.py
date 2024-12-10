from os import path, listdir
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torchvision.io import read_image
from torchvision.transforms import v2
from utils.unet_structure import UNet_Structure

class UnetTrainer(UNet_Structure):
    def __init__(self,
                 data_kwargs: dict,
                 optimizer_kwargs: dict,
                 num_epochs: int,
                 model_saving_path: str = None,
                 test_size: float = 0.25,
                 device: str = 'cpu') -> None:
        super().__init__(device=device)
        self.test_size = test_size
        self.num_epochs = num_epochs
        self.model_saving_path = model_saving_path
        self.batch_size = data_kwargs['batch_size']
        self.mean = data_kwargs['mean']
        self.std = data_kwargs['std']
        # Get loaders from disk
        self.training_loader, self.validation_loader = self._get_data_loader_from_disk(**data_kwargs)
        # Ensure that the function that got the dataloader correctly set sizes
        assert self.train_set_size != 0, 'The train set size has not been properly set'
        assert self.val_set_size != 0, 'The val set size has not been properly set'

        self.training_batch_number = int(self.train_set_size / data_kwargs['batch_size'])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        self.criterion = nn.functional.binary_cross_entropy_with_logits
        self.schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=(len(self.training_loader.dataset) * num_epochs) // self.training_loader.batch_size
        )
    # restore the model from a saved model
    def restore_model(self, model_path: str) -> None:
        model_save = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_save['model_state_dict'])
        self.model = self.model.to(torch.device(self.device))  # ensure model is on correct device
        self.optimizer.load_state_dict(model_save['optimizer_state_dict'])
        self.schedular.load_state_dict(model_save['schedular_state_dict'])
        self.cur_epoch = model_save['epoch']
        self.train_loss = model_save['train_loss']
        self.validation_loss = model_save['validation_loss']
        self.train_accuracy = model_save['train_accuracy']
        self.validation_accuracy = model_save['validation_accuracy']
        self.learning_rate = model_save['learning_rate']
        self.train_positive_f1_score = model_save['train_positive_f1_score']
        self.validation_positive_f1_score = model_save['validation_positive_f1_score']
        self.train_negative_f1_score = model_save['train_negative_f1_score']
        self.validation_negative_score = model_save['validation_negative_score']
        self.batch_size = model_save['batch_size']
        self.train_set_size = model_save['train_set_size']
        self.validation_set_size = model_save['validation_set_size']
        self.mean = model_save['mean']
        self.std = model_save['std']
        self.training_n_batch = model_save['training_n_batch']
    # Training the model
    def fit(self) -> None:
        for epoch in range(self.cur_epoch + 1, self.num_epochs + 1):
            self.cur_epoch += 1
            print(f"Start Training Epoch {epoch}...")
            t_loss, t_accuracy, t_positive_f1_score, t_negative_f1_score, lr = self._train_epoch()
            val_loss, val_acc, val_positive_f1_score, val_negative_f1_score = self._validate()
            print(
                f"- Average metrics: \n"
                f"\t\t- train loss={np.mean(t_loss):0.2e}, "
                f"train acc={np.mean(t_accuracy):0.3f}, "
                f"learning rate={np.mean(lr):0.3e} \n"
                f"\t\t- val loss={val_loss:0.2e}, "
                f"val acc={val_acc:0.3f} \n"
                f"Finish Training Epoch {epoch} !\n"
            )

            # Saving the metrics for later plotting
            self.training_loss.extend(t_loss)
            self.train_accuracy.extend(t_accuracy)
            self.learning_rate.extend(lr)
            self.train_positive_f1_score.extend(t_positive_f1_score)
            self.train_negative_f1_score.extend(t_negative_f1_score)
            self.validation_loss.append(val_loss)
            self.validation_accuracy.append(val_acc)
            self.validation_positive_f1_score.append(val_positive_f1_score)
            self.validation_negative_score.append(val_negative_f1_score)

            # Save the model if a saving path is provided
            if self.model_saving_path is not None:
                self._save_model()

        # Plot training curves
        self._plot_training_curves(self.num_epochs)

    @torch.no_grad()
    def _validate(self):
        """Compute the accuracy and loss for the validation set"""
        self.model.eval()

        test_loss = 0
        accuracy = 0
        pos_precision = 0
        pos_recall = 0
        pos_f1 = 0
        neg_precision = 0
        neg_recall = 0
        neg_f1 = 0

        for data, target in self.validation_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)

            batch_size = len(data)
            test_loss += self.criterion(output, target).item() * batch_size

            formatted_output = output.clamp(0, 1).round()
            acc, (p_p, r_p, f1_p), (p_n, r_n, f1_n) = get_metrics(formatted_output, target)

            accuracy += acc * batch_size
            pos_precision += p_p * batch_size
            pos_recall += r_p * batch_size
            pos_f1 += f1_p * batch_size
            neg_precision += p_n * batch_size
            neg_recall += r_n * batch_size
            neg_f1 += f1_n * batch_size

        test_loss /= self.val_set_size
        accuracy /= self.val_set_size
        pos_precision /= self.val_set_size
        pos_recall /= self.val_set_size
        pos_f1 /= self.val_set_size
        neg_precision /= self.val_set_size
        neg_recall /= self.val_set_size
        neg_f1 /= self.val_set_size

        return test_loss, accuracy, [pos_precision, pos_recall, pos_f1], [neg_precision, neg_recall, neg_f1]

    def _train_epoch(self):
        """Train one epoch"""
        self.model.train()

        loss_history = []
        accuracy_history = []
        prf1_p_history = []
        prf1_n_history = []
        lr_history = []

        # Compute array only once per epoch
        batch_to_print = np.linspace(0, self.training_batch_number, 5).round()

        for batch_idx, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.schedular.step()

            formatted_output = output.clamp(min=0, max=1).round()
            accuracy, (p_p, r_p, f1_p), (p_n, r_n, f1_n) = get_metrics(formatted_output, target)

            loss_history.append(loss.item())
            accuracy_history.append(accuracy)
            lr_history.append(self.schedular.get_last_lr()[0])
            prf1_p_history.append([p_p, r_p, f1_p])
            prf1_n_history.append([p_n, r_n, f1_n])

            if batch_idx in batch_to_print:
                print(
                    f"- Metrics of Batch {batch_idx:03d}/{self.training_batch_number}: \n"
                    f"\t\t- loss={loss.item():0.2e}, "
                    f"acc={accuracy:0.3f}, "
                    f"lr={self.schedular.get_last_lr()[0]:0.3e} "
                )

        return loss_history, accuracy_history, prf1_p_history, prf1_n_history, lr_history

    def _save_model(self):
        """Save important variables and model state"""
        state = {
            'epoch': self.cur_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'schedular_state_dict': self.schedular.state_dict(),
            'train_loss_history': self.training_loss,
            'train_acc_history': self.train_accuracy,
            'lr_history': self.learning_rate,
            'val_loss_history': self.validation_loss,
            'val_acc_history': self.validation_accuracy,
            'train_prf1_p_history': self.train_positive_f1_score,
            'train_prf1_n_history': self.train_negative_f1_score,
            'val_prf1_p_history': self.validation_positive_f1_score,
            'val_prf1_n_history': self.validation_negative_score,
            'batch_size': self.batch_size,
            'train_set_size': self.train_set_size,
            'val_set_size': self.val_set_size,
            'mean': self.mean,
            'std': self.std,
            'training_batch_number': self.training_batch_number
        }
        torch.save(state, path.join(self.model_saving_path, f'training_save_epoch_{self.cur_epoch}.tar'))

    def _get_data_loader_from_disk(self, batch_size: int, mean: torch.Tensor, std: torch.Tensor, img_folder: str,
                                   gt_folder: str):
        """Helper function to load data into train and val dataloader directly from disk"""

        class RoadDataset(torch.utils.data.Dataset):
            """Class representing our custom Dataset"""

            def __init__(self):
                self.imgs = list(sorted(listdir(img_folder)))
                self.gts = list(sorted(listdir(gt_folder)))

                self.to_float = v2.ToDtype(torch.float32, scale=True)
                self.transform_norm = v2.Compose([
                    self.to_float,
                    v2.Normalize(mean, std)
                ])

            def __getitem__(self, idx: int):
                img_path = path.join(img_folder, self.imgs[idx])
                gt_path = path.join(gt_folder, self.gts[idx])

                img = read_image(img_path)
                gt = read_image(gt_path)

                img = self.transform_norm(img)
                gt = self.to_float(gt).round()

                return img, gt

            def __len__(self):
                return len(self.imgs)

        dataset = RoadDataset()
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)

        split_idx = int(self.test_size * dataset_size)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split_idx:], indices[:split_idx]

        self.train_set_size = len(train_indices)
        self.val_set_size = len(val_indices)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                  pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, val_loader
