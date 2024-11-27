import os
import pdb
import time
import torch
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset

from MLProject.MLproject.src.datasets import BaseADDataset
from MLProject.MLproject.src.networks.SVDNetMnist import MLP_UAI
from MLProject.MLproject.src.optim.mnist_trainer import DeepOCCUAITrainer
from MLProject.MLproject.src.datasets.mnist_dataset import create_mnist_dataset

import MLProject.MLproject.src.datasets.mnist_dataset
from DeepOCC import DeepOCC  # Deep One-class Classification
from utils.config import Config
from datasets.odds import ODDSADDataset
import warnings
warnings.filterwarnings("ignore")


class CustomADDataset(BaseADDataset):
    def __init__(self, root, train_set, test_set):
        super().__init__(root)
        self.train_set = train_set
        self.test_set = test_set

    def loaders(self, batch_size, shuffle_train=True, shuffle_test=False, num_workers=0):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, test_loader

def main():
    anomaly_ratio = 0.2
    batch_size = 64
    n_epochs = 20
    lr = 0.001
    weight_decay = 1e-6
    nu = 0.05

    # Controlla se CUDA è disponibile, altrimenti utilizza la CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Creazione del dataset
    train_data, train_targets = create_mnist_dataset(anomaly_ratio=anomaly_ratio)

    # Dividere i dati in train e test
    num_train = int(0.8 * len(train_data))
    num_test = len(train_data) - num_train
    train_data, test_data = torch.utils.data.random_split(train_data, [num_train, num_test])
    train_targets, test_targets = torch.utils.data.random_split(train_targets, [num_train, num_test])

    # Convertire i subset in tensori
    train_data = torch.stack([train_data[i] for i in range(len(train_data))])
    test_data = torch.stack([test_data[i] for i in range(len(test_data))])
    train_targets = torch.stack([train_targets[i] for i in range(len(train_targets))])
    test_targets = torch.stack([test_targets[i] for i in range(len(test_targets))])

    # Creare TensorDataset dai dati e dai target
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)

    # Creare un oggetto CustomADDataset
    dataset = CustomADDataset(root='./data', train_set=train_dataset, test_set=test_dataset)

    # Inizializzare il modello con le dimensioni corrette
    model = MLP_UAI(x_dim=784, h_dims=[784, 128, 64], rep_dim=32).to(device)

    # Inizializzare il centro della ipersfera
    train_loader, _ = dataset.loaders(batch_size=batch_size, shuffle_train=True)
    c = DeepOCCUAITrainer.init_center_c(train_loader, model, device, 0.1)

    # Inizializzare il trainer
    trainer = DeepOCCUAITrainer(R=0.1, c=c, nu=nu, eta=1, n_epochs=n_epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, device=device)

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

if __name__ == "__main__":
    main()

