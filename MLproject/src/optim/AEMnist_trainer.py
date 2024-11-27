import pdb
import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from MLProject.MLproject.src.base.base_net import BaseNet
from MLProject.MLproject.src.base.base_trainer import BaseTrainer
from MLProject.MLproject.src.datasets.base_dataset import BaseADDataset
from MLProject.MLproject.src.datasets.mnist import MNISTADDDataset


class AEMnist_trainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
                 batch_size: int = 30, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, weight_decay, device, n_jobs_dataloader)

        # Results
        self.train_time = None

        self.test_auc = None
        self.test_time = None

    def train(self, dataset: MNISTADDDataset, ae_net: BaseNet):
        # Get train Esempi loader
        print('Starting pretrain')
        # Utilizzo un badget di 31 elementi
        train_loader, _ = dataset.loaders(batch_size=200, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Training
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):
            print(f'Epoch n. {epoch}')
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss = loss.item()
                n_batches += 1

            # if (epoch+1)%10 == 0:
            #     epoch_train_time = time.time() - epoch_start_time
            #     print('| Epoch: {:03}/{:03} | Train Time: {:.3f}s | Train Loss: {:.6f} |'.format(epoch+1, self.n_epochs, epoch_train_time, epoch_loss/n_batches))

        self.train_time = time.time() - start_time

        print('Pretraining Time: {:.3f}s'.format(self.train_time))
        print('Finished pretraining.')

        return ae_net

    def test(self, dataset: MNISTADDDataset, ae_net: BaseNet):
        # Get test Esempi loader


        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        self.labels = np.array(labels)
        self.scores = np.array(scores)
        '''# Convert labels to binary format: 0 for normal (2) and 1 for anomalies (0, 1, 3, 4, 5, 6, 7, 8, 9)
        labels_binary = np.where(np.isin(self.labels, [2]), 0, 1)

        # Handle one-vs-rest classification correctly
        scores_ndarray = self.scores.squeeze()  # Ensure scores are 1D if necessary

        # Handle binary classification'''

        self.test_auc = roc_auc_score(labels, scores, multi_class='ovr')
        # self.test_auc = roc_auc_score(labels, torch.tensor(self.scores).float().cpu().numpy().squeeze(), multi_class='ovr')

        print('Finished pretraining test.')
