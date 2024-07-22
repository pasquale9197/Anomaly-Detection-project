
import pdb
import time
import torch
import numpy as np
import torch.optim as optim
from itertools import product
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import DataLoader

from base.base_net import BaseNet
from base.base_trainer import BaseTrainer
from datasets.base_dataset import BaseADDataset

class DeepOCCUAITrainer(BaseTrainer):

    def __init__(self, R, c, nu, eta: float,
                 n_epochs: int = 150, batch_size: int = 128, weight_decay: float = 1e-6,
                 lr: float = 0.001, optimizer_name: str = 'adam',
                 device: str = 'cuda', n_jobs_dataloader: int = 0, al_loss=-1):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, weight_decay,
                         device, n_jobs_dataloader)

        self.R = torch.tensor(R, device=self.device)
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.eta = eta

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        self.eps = 1e-6

        self.al_loss = al_loss  # 0: reject, 1: reciprocal

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # for regression loss
        reg_loss = torch.nn.BCELoss()

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            self.c = self.init_center_c(train_loader, net)

        td_ascores = []  # for rbtd query strategy

        # Training
        # print('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)  # anomaly score

                if self.al_loss=='one_class_uai':  # L_uai(for labeled data) + L_base (for the entire data)
                    losses = dist.clone()

                elif self.al_loss=='soft_boundary_uai':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

                if 'soft' not in self.al_loss:
                    loss = torch.mean(losses)

                if self.al_loss=='one_class_uai' or self.al_loss=='soft_boundary_uai':
                    if len(semi_targets[semi_targets==1]) > 0 or len(semi_targets[semi_targets==-1]) > 0:
                        reg_outputs = net.forward_uai(outputs, dist)  # regression output

                        reg_normal = reg_outputs[semi_targets==1]  # normal: regression target -> 0
                        reg_abnormal = reg_outputs[semi_targets==-1]  # abnormal: regression target -> 1

                        reg_target = torch.cat([torch.zeros(reg_normal.shape), torch.ones(reg_abnormal.shape)])  #

                        uai_loss = reg_loss(torch.cat([reg_normal, reg_abnormal]), reg_target.to(self.device))
                        loss += uai_loss

                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.al_loss == 'soft_boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                if (self.al_loss == 'soft_boundary_nce') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                epoch_loss += loss.item()
                n_batches += 1

            self.test(dataset, net, silent=True)
            td_ascores.append(self.scores)  # save ascores at every epoch

            # if (epoch+1)%10 == 0:
            #     epoch_train_time = time.time() - epoch_start_time
            #     print('| Epoch: {:03}/{:03} | Train Time: {:.3f}s | Train Loss: {:.9f} |'.format(epoch+1, self.n_epochs, epoch_train_time, epoch_loss/n_batches))

        self.train_time = time.time() - start_time
        # print('Training Time: {:.3f}s'.format(self.train_time))
        # print('Finished training.')

        self.td_ascores = np.array(td_ascores)

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, silent=True):

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        # print('Starting testing...')
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

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = dist

                losses = dist
                loss = torch.mean(losses)

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            outputs.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores, outputs = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        outputs = np.array(outputs)

        self.scores = scores
        self.outputs = outputs
        if (dataset.ratio_pollution > 0) or (dataset.current_set_index == 'unseen'):
            self.test_auc = roc_auc_score(labels, scores)
        else:
            self.test_auc = -1

        if not silent:
            print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
            print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
            print('Test Time: {:.3f}s'.format(self.test_time))
            print('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
