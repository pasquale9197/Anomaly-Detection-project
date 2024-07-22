import pdb
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from base.base_net import BaseNet
from base.base_trainer import BaseTrainer
from datasets.base_dataset import BaseADDataset

class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, weight_decay, device, n_jobs_dataloader)

        # Results
        self.train_time = None

        self.test_auc = None
        self.test_time = None

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

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

                epoch_loss += loss.item()
                n_batches += 1

            # if (epoch+1)%10 == 0:
            #     epoch_train_time = time.time() - epoch_start_time
            #     print('| Epoch: {:03}/{:03} | Train Time: {:.3f}s | Train Loss: {:.6f} |'.format(epoch+1, self.n_epochs, epoch_train_time, epoch_loss/n_batches))

        self.train_time = time.time() - start_time

        # print('Pretraining Time: {:.3f}s'.format(self.train_time))
        # print('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        # Get test data loader
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

        if (dataset.ratio_pollution > 0) or (dataset.current_set_index == 'unseen'):
            self.test_auc = roc_auc_score(labels, scores)
        elif dataset.current_set_index == 'baseline':
            self.test_auc = roc_auc_score(labels, scores)
        else:
            self.test_auc = -1
        
        # print('Finished pretraining test.')
