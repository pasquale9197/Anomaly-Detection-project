import os
import sys
from abc import ABC
from pathlib import Path

import numpy
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.transforms import transforms


class Mnist_dataset(Dataset):

    def __init__(self, root: str, dataset_name: str, ratio_pollution=None, random_state=None, train=True):
        super(Dataset, self).__init__()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train=train
        if train:
            original_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        else:
            original_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.data = original_dataset.data
        self.target = original_dataset.targets
        self.targets = torch.where(self.target == 2, 1, 0)

        self.semi_targets = None
        self.global_index = None
        self.label_known = None
        self.ps_label_known = None

        self.x_normal = self.data[self.targets == 0]
        self.y_normal = self.targets[self.targets == 0]

        self.x_anomalous = self.data[self.targets == 1]
        self.y_anomalous = self.targets[self.targets == 1]

        self.n_normal = len(self.x_normal)
        self.n_outlier = len(self.x_anomalous)

        '''anomaly_class = 2

        x = mat['data'].T
        y = mat['label'][0]

        # Non utilizzando ratio_pollution = 0 non ho bisogno di un training sul dataset pulito senza anomalie
        x_train = np.concatenate([self.x_normal, self.x_anomalous], axis=0)
        y_train = np.concatenate([self.y_normal, self.y_anomalous], axis=0)

        #scaler = StandardScaler().fit(x_train.astype(np.float32))
        #x_train_std = scaler.transform(x_train.astype(np.float32))

        #minmax_scaler = MinMaxScaler().fit(x_train_std.astype(np.float32))
        #x_train_std = minmax_scaler.transform(x_train_std.astype(np.float32))


        self.targets = torch.where(self.targets == 2, 1, 0)

        self.test_set = torch.tensor(original_test_dataset, dtype=torch.float32)'''

        # self.data = torch.tensor(original_dataset, dtype=torch.float32)
        self.semi_targets = torch.zeros_like(self.targets)


    def __getitem__(self, index):
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        sample = transform(sample.numpy())

        if self.label_known is not None:
            label_known = self.label_known[index]
        else:
            label_known = None

        if self.ps_label_known is not None:
            ps_label_known = self.ps_label_known[index]
        else:
            ps_label_known = None

        if label_known == 1:
            if target == 1: # Abnormal
                semi_target = -1
            elif target == 0: # Normal
                semi_target = 1
            else:
                semi_target = 0
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def plot_samples(self, num_samples=10):
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
        for i in range(num_samples):
            index = np.random.randint(0, len(self.data))
            sample, target, _, _ = self.__getitem__(index)
            sample = sample.numpy().reshape(28, 28)
            axes[i].imshow(sample, cmap='gray')
            axes[i].set_title(f'Label: {target}')
            axes[i].axis('off')
        plt.show()

    def plot_normal_anomalous_samples(self, num_samples=5):
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))

        # Plot normal samples
        for i in range(num_samples):
            index = np.random.randint(0, len(self.x_normal))
            sample = self.x_normal[index].reshape(28, 28)
            axes[0, i].imshow(sample, cmap='gray')
            axes[0, i].set_title('Normal')
            axes[0, i].axis('off')

        # Plot anomalous samples
        for i in range(num_samples):
            index = np.random.randint(0, len(self.x_anomalous))
            sample = self.x_anomalous[index].reshape(28, 28)
            axes[1, i].imshow(sample, cmap='gray')
            axes[1, i].set_title('Anomalous')
            axes[1, i].axis('off')

        plt.show()