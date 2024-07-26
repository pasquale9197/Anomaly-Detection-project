import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

# Definizione di un modello semplice
class SimpleModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=1):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Caricamento del dataset MNIST
train_dataset = MNIST(root='./Esempi', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='./Esempi', train=False, download=True, transform=ToTensor())

# Preprocessing dei dati
train_data = train_dataset.data.view(-1, 28*28).float() / 255.0
train_targets = (train_dataset.targets != 0).float()  # Considera la classe 0 come normale e le altre come anomale

# Inizializzazione del modello, loss function e optimizer
model = SimpleModel().to('cpu')
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Addestramento del modello
train_loader = DataLoader(list(zip(train_data, train_targets)), batch_size=64, shuffle=True)
n_epochs = 5

for epoch in range(n_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')

# Testing del modello
test_data = test_dataset.data.view(-1, 28*28).float() / 255.0
test_targets = (test_dataset.targets != 0).float()

model.eval()
with torch.no_grad():
    test_output = model(test_data).squeeze().numpy()

# Calcolo delle metriche
test_preds = (test_output > 0.5).astype(int)
cm = confusion_matrix(test_targets, test_preds)
precision = precision_score(test_targets, test_preds)
recall = recall_score(test_targets, test_preds)
f1 = f1_score(test_targets, test_preds)
fpr, tpr, _ = roc_curve(test_targets, test_output)
auc = roc_auc_score(test_targets, test_output)

# Visualizzazione dei risultati
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'AUC: {auc:.4f}')

ConfusionMatrixDisplay(cm).plot()
plt.show()

plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
