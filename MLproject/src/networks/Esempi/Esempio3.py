import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Definizione della rete neurale DeepSVDDNet
class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim=784, rep_dim=32):
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
        return self.encoder(x)

# Funzione per inizializzare il centro della sfera
def init_center_c(train_loader, net, device):
    n_samples = 0
    c = torch.zeros((net.rep_dim,), device=device)

    net.eval()
    with torch.no_grad():
        for data, _ in train_loader:
            inputs = data.to(device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples
    c[(abs(c) < 1e-6) & (c < 0)] = -1e-6
    c[(abs(c) < 1e-6) & (c > 0)] = 1e-6

    return c

# Definizione del trainer per Deep SVDD
class DeepSVDDTrainer:
    def __init__(self, R, c, nu, n_epochs=150, lr=0.001, weight_decay=1e-6, device='cuda'):
        self.R = torch.tensor(R, device=device)
        self.c = torch.tensor(c, device=device).clone().detach() if c is not None else None
        self.nu = nu
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

    def train(self, dataset, net):
        train_loader, _ = dataset.loaders(batch_size=64, num_workers=0)
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.n_epochs):
            net.train()
            epoch_loss = 0.0
            n_batches = 0
            for data, _ in train_loader:
                inputs = data.to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {epoch_loss/n_batches:.4f}')

    def test(self, dataset, net):
        _, test_loader = dataset.loaders(batch_size=64, num_workers=0)
        net = net.to(self.device)

        epoch_loss = 0.0
        n_batches = 0
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data, labels in test_loader:
                inputs, labels = data, labels
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = dist

                loss = torch.mean(scores)
                epoch_loss += loss.item()
                n_batches += 1

                idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        print(f'Test Loss: {epoch_loss / n_batches:.4f}')
        return idx_label_score

# Funzione per creare un dataset di anomalie
def create_anomaly_dataset(dataset, normal_class=0, anomaly_ratio=0.1):
    train_data = dataset.data.float() / 255.0
    train_targets = dataset.targets

    normal_data = train_data[train_targets == normal_class]
    normal_targets = train_targets[train_targets == normal_class]

    anomaly_data = train_data[train_targets != normal_class]
    anomaly_targets = train_targets[train_targets != normal_class]

    num_anomalies = int(anomaly_ratio * len(normal_data))
    anomaly_data = anomaly_data[:num_anomalies]
    anomaly_targets = anomaly_targets[:num_anomalies]

    data = torch.cat((normal_data, anomaly_data), dim=0)
    targets = torch.cat((normal_targets, anomaly_targets), dim=0)

    return data, targets

# Classe CustomADDataset per gestire il dataset di addestramento e test
class CustomADDataset:
    def __init__(self, train_data, train_targets, test_data, test_targets):
        self.train_set = torch.utils.data.TensorDataset(train_data, train_targets)
        self.test_set = torch.utils.data.TensorDataset(test_data, test_targets)

    def loaders(self, batch_size, num_workers):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader

# Caricamento del dataset MNIST e preprocessing
train_dataset = MNIST(root='./Esempi', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='./Esempi', train=False, download=True, transform=ToTensor())

normal_class = 0
anomaly_ratio = 0.1

train_data, train_targets = create_anomaly_dataset(train_dataset, normal_class, anomaly_ratio)
test_data, test_targets = create_anomaly_dataset(test_dataset, normal_class, anomaly_ratio)

# Inizializzazione del dataset personalizzato
dataset = CustomADDataset(train_data.view(train_data.size(0), -1), train_targets,
                          test_data.view(test_data.size(0), -1), test_targets)

# Inizializzazione del modello e del trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepSVDDNet(input_dim=784, rep_dim=32).to(device)
R = 0.1  # Raggio iniziale dell'ipersfera
c = init_center_c(DataLoader(dataset.train_set, batch_size=64, shuffle=True), model, device)
trainer = DeepSVDDTrainer(R=R, c=c, nu=0.05, n_epochs=20, lr=0.001, weight_decay=1e-6, device=device)

# Addestramento del modello
trainer.train(dataset, model)

# Testing del modello
idx_label_score = trainer.test(dataset, model)

# Calcolo e visualizzazione dell'AUC
labels, scores = zip(*[(item[0], item[1]) for item in idx_label_score])
auc = roc_auc_score(labels, scores)
print(f'Test AUC: {auc:.4f}')

# Visualizzazione dell'istogramma dei punteggi di anomalia
plt.hist([score for label, score in zip(labels, scores) if label == 0], bins=50, alpha=0.5, label='Normali')
plt.hist([score for label, score in zip(labels, scores) if label == 1], bins=50, alpha=0.5, label='Anomali')
plt.xlabel('Punteggio di Anomalia')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei Punteggi di Anomalia')
plt.legend()
plt.show()
