import os
import pdb
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class ODDSDataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/
    """

    def __init__(self, root: str, dataset_name: str, ratio_pollution=None,
                 random_state=None, train=True):
        super(Dataset, self).__init__()

        max_rp_dict = {'ionosphere': 0.359,
                       'arrhythmia': 0.146,
                       'cardio': 0.096,
                       'mnist_tab': 0.092,
                       'glass': 0.042,
                       'optdigits': 0.029,
                       'nslkdd': 0.465}

        self.classes = [0, 1]  # 1: abnormal

        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.file_name = self.dataset_name + '.mat'
        self.data_file = self.root / self.file_name

        self.train = train

        self.semi_targets = None
        self.global_index = None
        self.label_known = None
        self.ps_label_known = None

        mat = loadmat(self.data_file.__str__())

        X = mat['X']
        y = mat['y'].ravel()
        idx_norm = y == 0
        idx_out = y == 1

        self.n_normal = sum(idx_norm)
        self.n_outlier = sum(idx_out)

        if ratio_pollution == 0:
            X_normal = X[idx_norm]
            y_normal = y[idx_norm]

            X_abnormal = X[idx_out]
            y_abnormal = y[idx_out]

            X_train = X_normal.copy()
            y_train = y_normal.copy()

            X_test = np.concatenate((X_normal, X_abnormal))
            y_test = np.concatenate((y_normal, y_abnormal))

            self.n_outlier = 0

        else:  # Transductive setting: training data == test data
            if ratio_pollution == max_rp_dict[self.dataset_name]:
                X_train = np.concatenate((X[idx_norm], X[idx_out]))
                y_train = np.concatenate((y[idx_norm], y[idx_out]))
            else:
                self.n_outlier = int((self.n_normal * ratio_pollution) / (1 - ratio_pollution))

                assert self.n_outlier >= 1, 'Unvalid ratio'

                idx_out_selected = np.random.choice(np.where(y == 1)[0], self.n_outlier)
                X_train = np.concatenate((X[idx_norm], X[idx_out_selected]))
                y_train = np.concatenate((y[idx_norm], y[idx_out_selected]))

            # X_test = X_train.copy()
            # y_test = y_train.copy()

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train.astype(np.float32))
        X_train_stand = scaler.transform(X_train.astype(np.float32))

        minmax_scaler = MinMaxScaler().fit(X_train_stand.astype(np.float32))
        X_train_stand = minmax_scaler.transform(X_train_stand.astype(np.float32))

        if ratio_pollution == 0:
            X_test_stand = scaler.transform(X_test.astype(np.float32))

        # Data and targets
        if self.train:
            self.data = torch.tensor(X_train_stand, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            if ratio_pollution == 0:
                self.data = torch.tensor(X_test_stand, dtype=torch.float32)
                self.targets = torch.tensor(y_test, dtype=torch.int64)
            else:
                self.data = torch.tensor(X_train_stand, dtype=torch.float32)
                self.targets = torch.tensor(y_train, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)  # 0: unlabeled

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
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
            if target == 1:  # Abnormal
                semi_target = -1
            elif target == 0:  # Normal
                semi_target = 1
            else:
                semi_target = 0

        if ps_label_known == 1:
            semi_target = -2

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)
