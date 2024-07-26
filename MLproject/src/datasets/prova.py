import time
import numpy as np
import torch
import sys
import os
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import DataLoader

# Ensure correct import paths
from mnist import MNISTADDDataset
from MLProject.MLproject.src.networks.SVDNetMnist import MLP_MNIST
from MLProject.MLproject.src.optim.mnist_trainer import DeepSVDMnistTrainer
from MLProject.MLproject.src.optim.mnist_trainer_obj import Mnist_trainer_obj

# Define dataset path
root = 'MLProject/MLproject/data/mnist-original.mat'  # Specify the correct path
dataset_name = 'mnist-original'

n_al_iter = 5
numQueriesRate = 0.01

# Load MNIST dataset
dataset = MNISTADDDataset(root=root, dataset_name=dataset_name)

# dataset.train_set.plot_samples()

# dataset.train_set.plot_normal_anomalous_samples(num_samples=5)


# Initialize neural network
x_dim = 28 * 28  # Input dimension for MNIST images 28x28
net = MLP_MNIST(x_dim)

aucs = []
total_z_centers = []
total_label_known = []
total_pseudo_abnormal = []
total_abs_boundary = []

total_dists = []  # (n_al_iter, n_data)
total_z_datas = []  # (n_al_iter, n_data, hidden_size)

# Trainer parameters
R = 1.0
c = None
nu = 0.1
eta = 1.0
n_epochs = 10
batch_size = 128
weight_decay = 1e-6
lr = 0.001
optimizer_name = 'adam'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_jobs_dataloader = 0
al_loss = 'one_class_uai'

for al_iter_idx in range(n_al_iter + 1):
    t_start_al_iter = time.time()

    trainer = Mnist_trainer_obj()
    trainer.set_network('MLP_MNIST')

    trainer.train(dataset, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs, batch_size=batch_size, weight_decay=weight_decay,
                  device=device, n_jobs_dataloader=n_jobs_dataloader, al_loss=al_loss)

    auc, dist, z_data = trainer.test(dataset=dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    aucs.append(auc)
    total_dists.append(dist)
    total_z_datas.append(z_data)
    #total_z_centers.append(Mnist_trainer_obj.trainer.c.cpu().numpy().copy())

    # Assuming dataset contains test data to visualize
    _, test_loader = dataset.loaders(batch_size=1, shuffle_test=False)

    # Convert distances to numpy array
    dist = np.array(dist)

    # Aggregate distances by taking the maximum across all classes
    aggregated_dist = np.max(dist, axis=1)

    # Debugging: Print shapes of dist and anomalies
    print(f"Shape of aggregated_dist: {aggregated_dist.shape}")

    # Experiment with different percentiles to adjust the threshold
    for percentile in [60, 75, 85]:
        threshold = np.percentile(aggregated_dist, percentile)
        anomalies = aggregated_dist > threshold
        print(f"Threshold at {percentile}th percentile: {threshold}, Number of anomalies detected: {np.sum(anomalies)}")

    # Choose the threshold to use (e.g., 95th percentile)
    threshold = np.percentile(aggregated_dist, 70)
    anomalies = aggregated_dist > threshold

    # Plot distance distribution with chosen threshold
    plt.figure(figsize=(10, 5))
    plt.hist(aggregated_dist, bins=50, alpha=0.7, label='Aggregated Distances')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.title('Distribution of Aggregated Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    print(f"Shape of anomalies: {anomalies.shape}")

    # Debugging: Print the number of anomalies detected
    print(f"Number of anomalies detected: {np.sum(anomalies)}")

    # Plotting
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()

    # Plot normal and anomaly examples
    normal_count = 0
    anomaly_count = 0

    for idx, (inputs, labels, semi_target, index) in enumerate(test_loader):  # Adjusted unpacking
        if normal_count < 5 and not anomalies[idx]:
            print(f"Normal Index: {idx}, Distance: {aggregated_dist[idx]}")
            axes[normal_count].imshow(inputs[0].cpu().numpy().reshape(28, 28), cmap='gray')
            axes[normal_count].set_title('Normal')
            axes[normal_count].axis('off')
            normal_count += 1

        if anomaly_count < 5 and anomalies[idx]:
            print(f"Anomaly Index: {idx}, Distance: {aggregated_dist[idx]}")
            axes[anomaly_count + 5].imshow(inputs[0].cpu().numpy().reshape(28, 28), cmap='gray')
            axes[anomaly_count + 5].set_title('Anomaly')
            axes[anomaly_count + 5].axis('off')
            anomaly_count += 1

        if normal_count >= 5 and anomaly_count >= 5:
            break

    plt.tight_layout()
    plt.show()

print('AUCs: ', aucs)
print('total_dists: ', total_dists)
