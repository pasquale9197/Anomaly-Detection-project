import os
import sys
from abc import ABC
from collections import defaultdict
from pathlib import Path

import numpy
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.transforms import transforms


class Mnist_dataset(Dataset):

    def __init__(self, root: str, dataset_name: str, ratio_pollution=None, random_state=None, train=True):
        super(Dataset, self).__init__()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.semi_targets = None
        self.global_index = None
        self.label_known = None
        self.ps_label_known = None

        # Loading the dataset
        if train:
            original_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        else:
            original_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)

        # Riduci il dataset
        num_anomalous_samples = 10
        normal_class = 2
        class_counts = defaultdict(int)
        selected_indices = []

        print()

        for idx, label in enumerate(original_dataset.targets):
            if label == normal_class or class_counts[label.item()] < num_anomalous_samples:
                selected_indices.append(idx)
                class_counts[label.item()] += 1

        reduced_data = Subset(original_dataset, selected_indices)
        reduced_labels = original_dataset.targets[selected_indices]

        transformed_data = []

        '''for subset_idx, (img, label) in enumerate(reduced_data):
            # Original image
            original_idx = selected_indices[subset_idx]
            original_img, original_label = original_dataset[original_idx]

            # Debug: Print original image before transformation
            plt.imshow(original_img, cmap='gray')
            plt.title(f'Original image at index {original_idx} with label {original_label}')
            plt.show()

            # Apply transformation
            transformed_img = transform(original_img)
            transformed_data.append(transformed_img)
            transformed_labels.append(original_label)

            # Debug: Print transformed image
            print(f'Transformed image at index {original_idx}:')
            print(transformed_img)'''

        for idx in selected_indices:
            img, label = original_dataset[idx]
            img = transform(img)  # Apply the transform manually
            transformed_data.append(img)

        self.data = transformed_data
        # self.data = [torch.randn(1, 28, 28) for _ in range(len(reduced_labels))]
        self.targets = reduced_labels
        self.target = reduced_labels
        self.targets = torch.where(self.targets == 2, torch.tensor(0), torch.tensor(1))

        self.n_normal = torch.sum(reduced_labels == normal_class).item()
        self.n_outlier = len(reduced_labels) - self.n_normal

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
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