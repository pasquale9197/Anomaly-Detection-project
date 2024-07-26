import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Definizione dell'Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Caricamento del dataset MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./Esempi', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./Esempi', train=False, transform=transform, download=True)

# Filtraggio delle classi per considerare solo una classe come normale (es. 0) e le altre come anomalie
normal_class = 0
anomaly_ratio = 0.1

train_data = train_dataset.data.float() / 255.0
train_targets = train_dataset.targets

normal_data = train_data[train_targets == normal_class]
normal_targets = train_targets[train_targets == normal_class]

anomaly_data = train_data[train_targets != normal_class]
anomaly_targets = train_targets[train_targets != normal_class]

num_anomalies = int(anomaly_ratio * len(normal_data))
anomaly_data = anomaly_data[:num_anomalies]
anomaly_targets = anomaly_targets[:num_anomalies]

train_data = torch.cat([normal_data, anomaly_data], dim=0)
train_targets = torch.cat([torch.zeros_like(normal_targets), torch.ones_like(anomaly_targets)], dim=0)

# Addestramento dell'Autoencoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

train_loader = DataLoader(list(zip(train_data, train_targets)), batch_size=64, shuffle=True)

num_epochs = 20
for epoch in range(num_epochs):
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Valutazione del modello
test_data = test_dataset.data.float() / 255.0
test_targets = test_dataset.targets
test_data = test_data.view(test_data.size(0), -1).to(device)

with torch.no_grad():
    outputs = autoencoder(test_data)
    loss = criterion(outputs, test_data)
    print(f'Test Loss: {loss.item():.4f}')

# Visualizzazione dei risultati
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_data[i].cpu().view(28, 28).numpy(), cmap='gray')
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(outputs[i].cpu().view(28, 28).numpy(), cmap='gray')
plt.show()

########################################################################################################################################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Definizione della rete neurale per Deep SVDD
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
    c = torch.zeros(net.rep_dim, device=device)

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

# Funzione per ottenere il raggio della sfera
def get_radius(dist: torch.Tensor, nu: float):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

# Trainer per Deep SVDD
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

# Inizializzazione
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepSVDDNet(input_dim=784).to(device)

# Caricamento del dataset MNIST e preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./Esempi', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./Esempi', train=False, transform=transform, download=True)

c = init_center_c(train_loader, model, device)
R = 0.0
trainer = DeepSVDDTrainer(R=R, c=c, nu=0.05, n_epochs=20, lr=0.001, weight_decay=1e-6, device=device)

# Addestramento del modello
trainer.train(train_dataset, model)
