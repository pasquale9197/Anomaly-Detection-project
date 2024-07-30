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
numQueriesRate = 0.01

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
n_epochs = 100
batch_size = 30
weight_decay = 1e-6
lr = 0.001
optimizer_name = 'adam'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def print_candidate_distribution(query_idx, dataset):
    labels = dataset.target[query_idx]  # Ottieni i label dei candidati
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
        ascore = dist.copy()

        aucs = np.append(aucs, auc)
        total_dist = np.append(total_dists, dist)
        total_z_datas = np.append(total_z_datas, z_data)
        # total_z_centers = np.append(total_z_centers, self.c)

        print('\tTest:: AUC: {:.4f}\t(Max (Stage {}): {:4f})\t(Stage 0: {:.4f})\n'.format(auc, np.argmax(aucs), max(aucs), aucs[0]))

        n_data = dataset.n_data
        numQueries = int(numQueriesRate * n_data)
        ps_numQueries = 0

        training_labels = dataset.train_set.targets

        candidate_idx = np.where(trainset_label_known == 0)[0]

        # candidate_idx = candidate_idx[candidate_idx < len(ascore)]
        # print(f"Candidate index size: {len(candidate_idx)}")

        if al_iter_idx == 0:
            qp = 0.8  # query point for adaptive linear
            qps = [qp]
            sortIdx = np.argsort(ascore)
            sortIdx = sortIdx[int(n_data * qp):int(n_data * qp) + numQueries]


            # valid_indices = np.arange(len(ascore))  # Modifica
            # valid_indices = valid_indices[int(len(ascore) * qp):int(len(ascore) * qp) + numQueries]
            # valid_indices = valid_indices[valid_indices < len(ascore)]  # Assicurarsi che gli indici siano validi
            # sortIdx = valid_indices

        else:
            n_qanomal = training_labels[queryIdx].sum().item()  # from the previous query information
            n_qnormal = len(queryIdx) - n_qanomal

            if n_qnormal + n_qanomal == 0:
                p_qnormal = 0  # Avoid division by zero
            else:
                p_qnormal = n_qnormal / (n_qnormal + n_qanomal)

            if qp == 1:  # qps[-1] == 1 (if the previous qp is 1)
                qp = (qp + qps[-2]) / 2
            else:
                qp = 2 * (1 - qp) * p_qnormal + (2 * qp - 1)
            qps.append(qp)

            # sortIdx = np.argsort(ascore[candidate_idx])

            if int(len(candidate_idx) * qp) + numQueries > len(sortIdx):  # Ensure it doesn't go out of bounds
                sortIdx = sortIdx[-numQueries:]
            else:
                sortIdx = sortIdx[int(len(candidate_idx) * qp):int(len(candidate_idx) * qp) + numQueries]
            sortIdx = sortIdx[sortIdx < len(candidate_idx)]  # Assicurarsi che gli indici siano validi
            sortIdx = candidate_idx[sortIdx[sortIdx < len(ascore)]]
        np.append(total_abs_boundary, qp)

        candidate = sortIdx[:numQueries]  # top-k
        if len(candidate) > 0:
            queryIdx = candidate
            trainset_label_known[queryIdx] = 1
        elif len(candidate) == 0:
            queryIdx = []

        np.append(total_label_known, trainset_label_known)
        #candidate_idx = np.array([])  # Azzeramento candidate_idx

        if ps_abnormal:
            trainset_ps_label_known = np.zeros(trainset_ps_label_known.shape)
            candidate_idx = np.where(trainset_label_known == 0)[0]
            labeled_idx = np.where(trainset_label_known == 1)[0]

            print(f"candidate_idx size: {len(candidate_idx)}, ascore size: {len(ascore)}")
            print(f"labeled_idx size: {len(labeled_idx)}")

            labeled_abnormal_idx = np.where(np.array(training_labels)[np.where(trainset_label_known == 1)[0]] == 1)[0]

            # Assicurarsi che gli indici siano validi
            candidate_idx = candidate_idx[candidate_idx < len(ascore)]
            labeled_idx = labeled_idx[labeled_idx < len(ascore)]

            # Debug: stampa le dimensioni dopo il filtraggio
            print(f"filtered candidate_idx size: {len(candidate_idx)}")
            print(f"filtered labeled_idx size: {len(labeled_idx)}")
            print(f"labeled_abnormal_idx size: {len(labeled_abnormal_idx)}")

            if len(labeled_abnormal_idx) == 0:
                candidate_idx = candidate_idx[ascore[candidate_idx] >= ascore[labeled_idx].max()]
                candidate_value_ascore = ascore[candidate_idx]
                sortIdx = np.argsort(candidate_value_ascore)
                ps_numQueries = numQueries
                sortIdx = candidate_idx[sortIdx[-ps_numQueries:]]
            else:
                '''labeled_abnormal_idx = labeled_idx[labeled_abnormal_idx]
                candidate_idx = candidate_idx[ascore[candidate_idx] >= np.median(ascore[labeled_abnormal_idx])]
                n_half_of_candidates = int(np.where(trainset_label_known == 0)[0].shape[0] * 0.5)
                if candidate_idx.shape[0] > n_half_of_candidates:
                    sortIdx = np.random.choice(candidate_idx, n_half_of_candidates)
                else:
                    sortIdx = candidate_idx.copy()'''
                labeled_abnormal_idx = labeled_abnormal_idx[labeled_abnormal_idx < len(ascore)]
                labeled_abnormal_idx = np.clip(labeled_abnormal_idx, 0, len(labeled_idx) - 1)
                labeled_abnormal_idx = labeled_idx[labeled_abnormal_idx]
                candidate_idx = candidate_idx[ascore[candidate_idx] >= np.median(ascore[labeled_abnormal_idx])]
                candidate_idx = candidate_idx[candidate_idx < len(ascore)]
                n_half_of_candidates = int(np.where(trainset_label_known == 0)[0].shape[0] * 0.5)
                if candidate_idx.shape[0] > n_half_of_candidates:
                    sortIdx = np.random.choice(candidate_idx, n_half_of_candidates)
                else:
                    sortIdx = candidate_idx.copy()


            trainset_ps_label_known[sortIdx] = 1

            np.append(total_pseudo_abnormal, trainset_ps_label_known.copy())

            n_known_normal = np.where(np.array(training_labels)[np.where(trainset_label_known == 1)[0]] == 0)[0].size
            n_known_outlier = np.where(np.array(training_labels)[np.where(trainset_label_known == 1)[0]] == 1)[0].size
            n_known_normal_new = np.where(np.array(training_labels)[queryIdx] == 0)[0].size
            n_known_outlier_new = np.where(np.array(training_labels)[queryIdx] == 1)[0].size
            print('\tAL step {} DONE. Query target --> [Normal: {}/ Abnormal: {} + {}]'.format(al_iter_idx,
                                                                                               n_known_normal,
                                                                                               n_known_outlier,
                                                                                               int(sum(trainset_ps_label_known))))
            print('\t (new) [Normal: {}/ Abnormal: {}]'.format(n_known_normal_new, n_known_outlier_new))

            dataset.update_label_known(trainset_label_known, queryIdx)
            dataset.update_ps_label_known(trainset_ps_label_known)
            print('\tAL Stage time: {:.3f}s'.format(time.time() - t_start_al_iter))

            # print_candidate_distribution(queryIdx, dataset)

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

        results = {}
        results['aucs'] = aucs

        results['test_targets'] = training_labels
        results['total_label_known'] = total_label_known
        results['total_pseudo_abnormal'] = total_pseudo_abnormal

        results['total_dists'] = total_dists
        results['total_z_datas'] = total_z_datas
        results['total_z_centers'] = total_z_centers

        results['total_abs_boundary'] = total_abs_boundary

