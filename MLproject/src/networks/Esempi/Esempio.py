import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
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

num_anomalies = int(anomaly_ratio * len(anomaly_data))
anomaly_data = anomaly_data[:num_anomalies]
anomaly_targets = anomaly_targets[:num_anomalies]

train_data = torch.cat([normal_data, anomaly_data], dim=0)
train_targets = torch.cat([torch.zeros_like(normal_targets), torch.ones_like(anomaly_targets)], dim=0)

# Addestramento dell'Autoencoder
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
