import sys
import os
import time

import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn

from MLProject.MLproject.src.base import BaseNet
from MLProject.MLproject.src.networks.SVDNetMnist import CustomADDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caricamento del dataset MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./Esempi', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./Esempi', train=False, transform=transform, download=True)


def create_anomaly_dataset(dataset, normal_class, anomaly_ratio=0.1):
    normal_idx = [i for i, (img, label) in enumerate(dataset) if label == normal_class]
    anomaly_idx = [i for i, (img, label) in enumerate(dataset) if label != normal_class]

    np.random.shuffle(anomaly_idx)
    num_anomalies = int(len(normal_idx) * anomaly_ratio)
    selected_anomaly_idx = anomaly_idx[:num_anomalies]

    normal_labels = [0] * len(normal_idx)
    anomaly_labels = [1] * num_anomalies

    data_idx = normal_idx + selected_anomaly_idx
    data_labels = normal_labels + anomaly_labels

    dataset.data = dataset.data[data_idx]
    dataset.targets = torch.tensor(data_labels)

    return dataset


normal_class = 0
anomaly_ratio = 0.1  # parametro s
train_dataset = create_anomaly_dataset(train_dataset, normal_class, anomaly_ratio)
test_dataset = create_anomaly_dataset(test_dataset, normal_class, anomaly_ratio)


class CustomMnistDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.semi_targets = torch.zeros(len(dataset))  # Semi targets inizializzati a zero

    def __getitem__(self, index):
        img, target = self.dataset[index]
        semi_target = self.semi_targets[index]  # Ottieni il semi target
        img = img.view(-1).float()  # Appiattisci l'immagine
        return img, target, semi_target, index

    def __len__(self):
        return len(self.dataset)



# Crea i custom dataset che restituiscono anche i semi_targets e gli indici
train_dataset = CustomMnistDataset(train_dataset)
test_dataset = CustomMnistDataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def apply_pca(data, n_components=50):
    pca = PCA(n_components=n_components)
    data_flat = data.view(data.size(0), -1).numpy()
    data_pca = pca.fit_transform(data_flat)
    return torch.tensor(data_pca, dtype=torch.float32)


# Applica PCA ai dati di addestramento e di test
train_data_pca = apply_pca(train_dataset.dataset.data, n_components=50)
test_data_pca = apply_pca(test_dataset.dataset.data, n_components=50)

train_dataset.dataset.data = train_data_pca
test_dataset.dataset.data = test_data_pca


class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim, rep_dim=32):
        super(DeepSVDDNet, self).__init__()
        self.rep_dim = rep_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, rep_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Appiattisci i dati
        return self.encoder(x)


def init_center_c(train_loader, net, device, eps=0.1):
    n_samples = 0
    c = torch.zeros((net.rep_dim,), device=device)  # Correzione qui

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            inputs = data[0].float().to(device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist, nu):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


from MLProject.MLproject.src.base.base_trainer import BaseTrainer
from MLProject.MLproject.src.datasets.base_dataset import BaseADDataset


class DeepOCCUAITrainer(BaseTrainer):

    def __init__(self, R, c, nu, eta: float,
                 n_epochs: int = 150, batch_size: int = 128, weight_decay: float = 1e-6,
                 lr: float = 0.001, optimizer_name: str = 'adam',
                 device: str = 'cuda', n_jobs_dataloader: int = 0, al_loss=-1):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, weight_decay,
                         device, n_jobs_dataloader)

        self.R = torch.tensor(R, device=self.device)
        self.c = torch.tensor(c, device=self.device).clone().detach() if c is not None else None
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

        # Get train Esempi loader
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
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                print(data)  # Stampa il contenuto dei dati per debug
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)  # anomaly score

                if self.al_loss=='one_class_uai':  # L_uai(for labeled Esempi) + L_base (for the entire Esempi)
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

        self.train_time = time.time() - start_time
        self.td_ascores = np.array(td_ascores)

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, silent=True):

        # Get test Esempi loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
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
        """Initialize hypersphere center c as the mean from an initial forward pass on the Esempi."""
        n_samples = 0
        c = torch.zeros((net.rep_dim,), device=self.device)

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

# Inizializzazione
model = DeepSVDDNet(input_dim=50).to(device)
c = init_center_c(train_loader, model, device)
R = 0.0

# Definizione del trainer
trainer = DeepOCCUAITrainer(R=R, c=c, nu=0.05, eta=1, n_epochs=20, batch_size=64, lr=0.001, weight_decay=1e-6, device=device)

# Definizione del dataset
dataset = CustomADDataset(root='./Esempi', train_set=train_dataset, test_set=test_dataset)

# Addestramento del modello
trainer.train(dataset, model)

