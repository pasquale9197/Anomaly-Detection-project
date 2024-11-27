import time
import numpy as np
import torch
import sys
import os
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import DataLoader
import tensorflow

from MLProject.MLproject.src.networks import MLP_Autoencoder
# Ensure correct import paths
from mnist import MNISTADDDataset
from MLProject.MLproject.src.networks.SVDNetMnist import MLP_MNIST
from MLProject.MLproject.src.optim.mnist_trainer import DeepSVDMnistTrainer
from MLProject.MLproject.src.optim.mnist_trainer_obj import Mnist_trainer_obj

# Define dataset path
root = 'MLProject/MLproject/data/mnist-original.mat'  # Specify the correct path
dataset_name = 'mnist-original'
n_al_iter = 15
numQueriesRate = 0.0065

# Initialize neural network
x_dim = 28 * 28  # Input dimension for MNIST images 28x28
net = MLP_MNIST(x_dim)
n_repeat = 5
total_dists = []  # (n_al_iter, n_data)
total_z_datas = []  # (n_al_iter, n_data, hidden_size)
ps_abnormal = True
pretrain = True

# Trainer parameters
R = 1.0
c = None
nu = 0.1
eta = 1.0
n_epochs = 50
batch_size = 30
weight_decay = 1e-6
lr = 0.001
optimizer_name = 'adam'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def print_candidate_distribution(candidate_idx, dataset):
    labels = dataset.target[candidate_idx]  # Ottieni i label dei candidati
    numbers, counts = np.unique(labels, return_counts=True)
    specific_numbers = [1, 2,  3, 4, 5, 6, 7, 8, 9]
    specific_counts = {num: 0 for num in specific_numbers}


    for number, count in zip(numbers, counts):
        if number in specific_counts:
            specific_counts[number] = count


    print("Distribuzione dei candidati selezionati:")
    for number in specific_numbers:
        print(f"Numero {number}: {specific_counts[number]} occorrenze")

    # Prepara i dati per il plot
    x = list(specific_counts.keys())
    y = list(specific_counts.values())

    # Histogram
    plt.bar(x, y, color='green')
    plt.xlabel('Numbers')
    plt.ylabel('Counts')
    plt.title('Histogram of Candidate Distribution')

    plt.tight_layout()
    plt.show()


# Check if CUDA is available and print device information
print("CUDA available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version: ", torch.version.cuda)
    print("CUDA device count: ", torch.cuda.device_count())
    print("CUDA current device: ", torch.cuda.current_device())
    print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("GPU not detected.")


n_jobs_dataloader = 0
al_loss = 'soft_boundary_uai'

for n_repeat_idx in range(n_repeat):
    print('N_REPEAT: ', n_repeat_idx)
    aucs = []  # (n_al_iter,)

    total_dists = []  # (n_al_iter, n_data)
    total_z_datas = []  # (n_al_iter, n_data, hidden_size)
    total_z_centers = []  # (n_al_iter, hidden_size)

    total_label_known = []  # (n_al_iter, n_data)
    total_pseudo_abnormal = []  # (n_al_iter, n_data)

    total_abs_boundary = []  # (n_al_iter)

    # Load MNIST dataset
    dataset = MNISTADDDataset(root=root, dataset_name=dataset_name)
    labels, counts = np.unique(dataset.targets, return_counts=True)
    print(f"Labels: {labels}, Counts: {counts}")

    # Initialization
    trainset_label_known = dataset.targets
    trainset_ps_label_known = np.zeros(dataset.n_data)  # pseudo label

    n_known_normal = 0  # number of quried normal Esempi
    n_known_outlier = 0
    trainig_labels_viewd = torch.zeros_like(dataset.train_set.targets)
    qp = 0.5
    print(
    '\t##########################################################################################################\n',
    '########################################### INIZIO TRAINING ##############################################\n',
    '##########################################################################################################')
    for al_iter_idx in range(n_al_iter + 1):
        t_start_al_iter = time.time()
        trainer = Mnist_trainer_obj()
        trainer.set_network('MLP_MNIST')
        if pretrain:  ## Pretrain model on datasets (via autoencoder)
            trainer.pretrain(dataset, optimizer_name=optimizer_name, lr=lr, n_epochs=50, batch_size=batch_size,
                             weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)

        trainer.train(dataset, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs, batch_size=50, weight_decay=weight_decay,
                  device=device, n_jobs_dataloader=n_jobs_dataloader, al_loss=al_loss)

        auc, dist, z_data = trainer.test(dataset=dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Copy distances to a new variable ascore
        ascore = dist.copy()

        # Append the current AUC score to the aucs array
        aucs = np.append(aucs, auc)
        # Append the current distances to the total_dists array
        total_dist = np.append(total_dists, dist)
        # Append the current z_data to the total_z_datas array
        total_z_datas = np.append(total_z_datas, z_data)
        # Commented out: Append the center c to the total_z_centers array
        # total_z_centers = np.append(total_z_centers, self.c)

        # Print the current test AUC, max AUC at any stage, and the AUC at stage 0
        print(
            '\tTest:: AUC: {:.4f}\t(Max (Stage {}): {:4f})\t(Stage 0: {:.4f})\n'.format(auc, np.argmax(aucs), max(aucs),
                                                                                        aucs[0]))

        # Get the number of data points in the dataset
        n_data = dataset.n_data
        # Calculate the number of queries based on the query rate
        numQueries = int(numQueriesRate * n_data)
        # Initialize the number of pseudo queries to 0
        ps_numQueries = 0

        # Get the training labels from the dataset
        training_labels = dataset.train_set.targets

        # Find indices of samples that have not been labeled (trainset_label_known is 0)
        # candidate_idx = np.where(trainset_label_known == 0)[0]
        # qp = 0.5  # query point for adaptive linear

        # First iteration of active learning
        if al_iter_idx == 0:
            # Set the query point for adaptive linear selection
            # Initialize the list of query points
            qps = [qp]
            # Sort the indices of ascore in ascending order
            candidate_idx = np.argsort(ascore)
            # Select a slice of sorted indices starting from the position determined by qp
            bound = int(len(candidate_idx) * qp)
            print()
            sortIdx = candidate_idx[bound:bound + numQueries]
            trainig_labels_viewd[sortIdx] = 1
            n_qanomal = training_labels[sortIdx].sum().item()
            n_qnormal = len(sortIdx) - n_qanomal
            n_known_normal += n_qnormal
            n_known_outlier += n_qanomal

        else:
            candidate_idx = np.argsort(ascore)
            bound = int(len(candidate_idx) * qp)
            mask = trainig_labels_viewd == 0
            sortIdxtmp = []
            sortIdx = (candidate_idx[bound:bound + numQueries])
            for l1 in sortIdx:
                if trainig_labels_viewd[l1] == 1:
                    continue
                else:
                    sortIdxtmp = np.append(sortIdxtmp, l1)
                    trainig_labels_viewd[l1] = 1
            sortIdx = sortIdxtmp
            n_qanomal = training_labels[sortIdx].sum().item()
            n_qnormal = len(sortIdx) - n_qanomal
            n_known_normal += n_qnormal
            n_known_outlier += n_qanomal

            if int(len(candidate_idx) * qp) + numQueries > len(sortIdx):  # n_data:
                # print('reach Max ascore')
                sortIdx = sortIdx[-numQueries:]
            else:
                sortIdx = sortIdx[int(n_data * qp):int(n_data * qp) + numQueries]

        print('   =========================================== DONE ===========================================')

        # ''' ================================== '''
        # ''' == Save results & Visualization == '''
        # ''' ================================== '''
        aucs = np.array(aucs)  # (n_al_iters,)

        test_targets = np.array(training_labels)
        total_label_known = np.array(total_label_known)  # (n_al_iters, n_data)
        total_pseudo_abnormal = np.array(total_pseudo_abnormal)  # when ps_abnormal is True

        total_dists = np.array(total_dists)  # (n_al_iters, n_data)
        total_z_datas = np.array(total_z_datas)  # (n_al_iters, n_data, h_dim)
        total_z_centers = np.array(total_z_centers)  # (n_al_iters, h_dim)

        total_abs_boundary = np.array(total_abs_boundary)

        print(f'Step: {al_iter_idx+1} di {n_al_iter}. Anomalie trovate: {n_qanomal} '
              f'su {n_qanomal+n_qnormal}.\n \n Totale dati esaminati: {n_known_normal}, totale anomalie: {n_known_outlier}')

        results = {}
        results['aucs'] = aucs

        results['test_targets'] = training_labels
        results['total_label_known'] = total_label_known
        results['total_pseudo_abnormal'] = total_pseudo_abnormal

        results['total_dists'] = total_dists
        results['total_z_datas'] = total_z_datas
        results['total_z_centers'] = total_z_centers

        results['total_abs_boundary'] = total_abs_boundary
        print_candidate_distribution(sortIdx, dataset)




