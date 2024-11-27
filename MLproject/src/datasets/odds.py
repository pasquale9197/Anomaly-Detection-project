import pdb
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from .base_dataset import BaseADDataset
from .odds_dataset import ODDSDataset

class ODDSADDataset(BaseADDataset):

    def __init__(self, root: str, dataset_name: str, ratio_pollution: float = 0.0,
                 random_state=None, label_known=None, ps_label_known=None):

        super().__init__(root)

        self.dataset_name = dataset_name
        self.ratio_pollution = ratio_pollution

        self.train_set = ODDSDataset(root=self.root, dataset_name=dataset_name, ratio_pollution=self.ratio_pollution,
                                    random_state=random_state, train=True)

        self.un = self.train_set.n_normal
        self.ua = self.train_set.n_outlier
        self.n_data = self.un + self.ua

        if not label_known:
            self.label_known = np.zeros((self.n_data))
        else:
            self.label_known = label_known

        if not ps_label_known:
            self.ps_label_known = np.zeros((self.n_data))
        else:
            self.ps_label_known = ps_label_known

        self.train_set.label_known = self.label_known
        self.train_set.ps_label_known = self.ps_label_known

        # for NCE
        self.idx_labeled_normal = []
        self.idx_labeled_abnormal = []
        self.idx_pseudo_abnormal = []

        if ratio_pollution == 0:
            self.test_set = ODDSDataset(root=self.root, dataset_name=dataset_name, ratio_pollution=self.ratio_pollution,
                                        random_state=random_state, train=False)  # the entire Esempi
            self.tn = np.where(self.test_set.targets==0)[0].__len__()
            self.ta = np.where(self.test_set.targets==1)[0].__len__()

    def update_label_known(self, label_known, queryIdx):
        self.label_known = label_known
        self.train_set.label_known = label_known

        if queryIdx.__len__() > 0:
            flag_for_abnormal = self.train_set.targets[queryIdx]==1  # torch.ByteTensor
            self.idx_labeled_abnormal += queryIdx[flag_for_abnormal.type(torch.bool)].tolist()
            self.idx_labeled_normal += queryIdx[(~flag_for_abnormal).type(torch.bool)].tolist()

    def update_ps_label_known(self, ps_label_known):
        self.ps_label_known = ps_label_known
        self.train_set.ps_label_known = ps_label_known

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        if self.ratio_pollution == 0:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers, drop_last=True)
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers, drop_last=False)
        else:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=num_workers, drop_last=True)
            test_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_test,
                                        num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
