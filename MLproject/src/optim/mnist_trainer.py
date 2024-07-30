import time
import torch
import sys
import os
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import DataLoader

from MLProject.MLproject.src.base.base_net import BaseNet
from MLProject.MLproject.src.base.base_trainer import BaseTrainer
from MLProject.MLproject.src.datasets import BaseADDataset
from MLProject.MLproject.src.datasets.mnist import MNISTADDDataset


class DeepSVDMnistTrainer(BaseTrainer):

    def __init__(self, R, c, nu, eta: float, n_epochs: int = 150, batch_size: int = 128,
                 weight_decay: float = 1e-6, lr: float = 0.001, optimizer_name: str = 'adam',
                 device: str = 'cuda', n_jobs_dataloader: int = 0, al_loss=-1):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, weight_decay, device, n_jobs_dataloader)

        self.R = torch.tensor(R, device=self.device)
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.eta = eta

        # Optimization parameters
        self.warm_up_n_epochs = 10
        self.eps = 1e-6

        self.al_loss = al_loss

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: MNISTADDDataset, net: BaseNet):
        train_loader, _ = dataset.loaders(batch_size=200, num_workers=self.n_jobs_dataloader)
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        reg_loss = torch.nn.BCELoss()
        if self.c is None or torch.isnan(self.c).any:
            self.c = self.init_center_c(train_loader, net)
        td_ascores = []

        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, targets, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                optimizer.zero_grad()
                rep, outputs = net(inputs)
                dist = torch.sum((rep - self.c) ** 2, dim=1)

                if self.al_loss == 'one_class_uai':
                    losses = dist.clone()

                elif self.al_loss=='soft_boundary_uai':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 /self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

                if 'soft' not in self.al_loss:
                    loss = torch.mean(losses)

                if self.al_loss=='one_class_uai' or self.al_loss=='soft_boundary_uai':
                    if len(semi_targets[semi_targets==1]) > 0 or len(semi_targets[semi_targets==-1]) > 0:
                        # reg_outputs = net.forward_uai(outputs, dist)
                        s = torch.unsqueeze(dist, 1)
                        concat = torch.cat(([rep, s]), dim=1)
                        reg_outputs = net.output_activation(net.reg(concat))

                        reg_normal = reg_outputs[semi_targets == 1]
                        reg_abnormal = reg_outputs[semi_targets == -1]

                        reg_target = torch.cat([torch.zeros(reg_normal.shape), torch.ones(reg_abnormal.shape)])

                        uai_loss = reg_loss(torch.cat([reg_normal, reg_abnormal]), reg_target.to(self.device))
                        loss += uai_loss

                loss.backward()
                optimizer.step()

                if (self.al_loss == 'soft_boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                if(self.al_loss == 'soft_boundary_nce') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                epoch_loss = loss.item()
                n_batches += 1

            self.test(dataset, net, silent=False)
            td_ascores.append(self.scores)
            
        self.train_time = time.time() - start_time
        self.td_ascores = np.array(td_ascores)

        return net

    def test(self, dataset: MNISTADDDataset, net: BaseNet, silent=False):
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        net = net.to(self.device)

        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                rep, outputs = net(inputs)
                if self.c is None or torch.isnan(self.c).any:
                    self.c = torch.mean(rep, dim=0)
                dist = torch.sum((rep - self.c) ** 2, dim=1)
                # scores = torch.softmax(outputs, dim=1) if outputs.size(1) > 1 else outputs.squeeze()
                scores = dist
                losses = torch.sum((rep - self.c) ** 2, dim=1)
                loss = torch.mean(losses)

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            outputs.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1
        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        _, labels, scores, outputs = zip(*idx_label_score)
        self.outputs = np.array(outputs)
        self.labels = np.array(labels)
        self.scores = np.array(scores)

        '''# Convert labels to binary format: 0 for normal (2) and 1 for anomalies (0, 1, 3, 4, 5, 6, 7, 8, 9)
        labels_binary = np.where(np.isin(self.labels, [2]), 0, 1)
        # Handle one-vs-rest classification correctly
        scores_ndarray = self.scores.squeeze()  # Ensure scores are 1D if necessary

        # Handle binary classification'''

        self.test_auc = roc_auc_score(labels, scores, multi_class='ovr')
        if not silent:
                print('Test Loss {:.6f}'.format(epoch_loss/n_batches))
                print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
                print('Test Time: {:.3f}s'.format(self.test_time))
                print('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs, _ = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

