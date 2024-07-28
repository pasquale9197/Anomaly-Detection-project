import os
import pdb
import time
import click
import torch
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from DeepOCC import DeepOCC  # Deep One-class Classification
from utils.config import Config
from datasets.odds import ODDSADDataset

import warnings
warnings.filterwarnings("ignore")

''' python main.py [dataset_name] [ratio_pollution] [al_loss] [qs_type] [ps_abnormal]
'''

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['nslkdd', 'ionosphere', 'arrhythmia',
                                                   'cardio', 'mnist_tab', 'glass', 'optdigits']))
@click.argument('ratio_pollution', type=float)  # Contamination ratio of unlabeled training datasets.

@click.argument('al_loss', type=str)  # 'soft_boundary': Reject (soft-boundary Deep SVDD)
                                      # 'one_class': Reject (one-class Deep SVDD)
                                      # 'dsad': Reject + Reciprocal (Deep SAD)
                                      # 'nce': Reject (one-class Deep SVDD) + NCE
                                      # 'soft_boundary_nce': Reject (soft-boundary Deep SVDD) + NCE
                                      # 'one_class_uai': UAI applied (+MLP)
                                      # 'soft_boundary_uai'
@click.argument('qs_type', type=str)  # 'random': random
                                      # 'mlp': most-likely positive (high confidence)
                                      # 'db': decision boundary
                                      # 'abs': adaptive boundary search
                                      # 'no': No oracle
@click.argument('ps_abnormal', type=int)  # pseudo-abnormal 0: not use, 1: use

@click.option('--seed', type=int, default=0, help='Set seed.')
@click.option('--n_repeat', type=int, default=5, help='Number of random seeds for training & evaluation.')

@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep OCC network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep OCC network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')  
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep OCC objective.')

@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining.')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')

@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for Esempi loading. 0 means that the Esempi will be loaded in the main process.')


def main(dataset_name, ratio_pollution, al_loss, qs_type, ps_abnormal, seed, n_repeat,
         optimizer_name, lr, n_epochs, batch_size, weight_decay,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_batch_size, ae_weight_decay,
         device, num_threads, n_jobs_dataloader):

    model_name = 'AL-L{}-QS{}-PS{}'.format(al_loss, qs_type, ps_abnormal)

    n_al_iter = 5
    numQueriesRate = 0.01

    ratio_pollution_dict = {'ionosphere':0.359,
                            'arrhythmia':0.146,
                                'cardio':0.096,
                             'mnist_tab':0.092,
                                 'glass':0.042,
                             'optdigits':0.029,
                            'nslkdd':0.465}

    if ratio_pollution > ratio_pollution_dict[dataset_name]:
        ratio_pollution = ratio_pollution_dict[dataset_name]
        print('Ratio pollution has been changed to Max rate.')

    print('\n\n\t----- Experiment Setting -----')
    print('\t                   Model: {}'.format(model_name))
    print('\t                 Dataset: {}'.format(dataset_name))
    print('\t         Ratio pollution: {:03.1f}%\n'.format(ratio_pollution*100))
    print('\t               # Repeats: {}'.format(n_repeat))

    net_name = '{}_mlp'.format(dataset_name)
    if 'uai' in al_loss:
        net_name = net_name +'_uai'
    folder_tag = 'p{:.1f}'.format(ratio_pollution*100)

    # Network parameter
    if dataset_name == 'nslkdd':
        ae_batch_size = 1024
        batch_size = 1024

    # Get configuration
    cfg = Config(locals().copy())

    # Path setting
    data_path = '../data'
    result_path = '../Results/{}/Results_{}_{}'.format(model_name, dataset_name, folder_tag)
    if not os.path.isdir('../Results/{}'.format(model_name)):
        os.mkdir('../Results/{}'.format(model_name))
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'

    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)


    ############################################################################
    # Main
    ############################################################################
    for n_repeat_idx in range(n_repeat):  # for random seeds

        print('\n')
        print('========================================')
        print('== #Repeat(seed): {:3d}/{:3d} =='.format(n_repeat_idx+1, n_repeat))
        print('========================================')

        # Set seed
        cfg.settings['seed'] = n_repeat_idx+1  # 1, 2, 3, 4, 5

        if cfg.settings['seed'] != -1:
            random.seed(cfg.settings['seed'])
            np.random.seed(cfg.settings['seed'])
            torch.manual_seed(cfg.settings['seed'])
            torch.cuda.manual_seed(cfg.settings['seed'])
            torch.backends.cudnn.deterministic = True

        # for results (.pickle) for every repeat
        aucs = []                   # (n_al_iter,)

        total_dists = []            # (n_al_iter, n_data)
        total_z_datas = []          # (n_al_iter, n_data, hidden_size)
        total_z_centers = []        # (n_al_iter, hidden_size)

        total_label_known = []      # (n_al_iter, n_data)
        total_pseudo_abnormal = []  # (n_al_iter, n_data)

        total_abs_boundary = []     # (n_al_iter)

        # DATA ##################################################################
        dataset = ODDSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                ratio_pollution=ratio_pollution,
                                random_state=cfg.settings['seed'])


        print('[TR] # Unlabeled   Normal: {}/{}'.format(dataset.un, dataset.un+dataset.ua))
        print('[TR] # Unlabeled Abnormal: {}/{} ({:.3f}%)'.format(dataset.ua, dataset.un+dataset.ua, dataset.ua/(dataset.un+dataset.ua)*100))

        # Initialization
        trainset_label_known = dataset.train_set.label_known
        trainset_ps_label_known = dataset.train_set.ps_label_known  # pseudo label

        n_known_normal = 0  # number of quried normal Esempi
        n_known_outlier = 0

        ####################################################################
        for al_iter_idx in range(n_al_iter+1):
            t_start_al_iter = time.time()

            print('\n\n\n== [Repeat: {}/{}] Active learning stage #{}/{}\t[Labeled: {:.2f}%] (Normal: {} / Abnormal: {}) =='.format(n_repeat_idx+1, n_repeat, al_iter_idx, n_al_iter, \
                                                                            (n_known_normal+n_known_outlier)/(dataset.un+dataset.ua)*100, n_known_normal, n_known_outlier))

            ''' ==================== '''
            ''' == Model Training == '''
            ''' ==================== '''

            # Initialize deepOCC model and set neural network phi
            deepOCC = DeepOCC()
            deepOCC.set_network(net_name)

            if pretrain:  ## Pretrain model on datasets (via autoencoder)
                deepOCC.pretrain(dataset,
                                    optimizer_name=cfg.settings['ae_optimizer_name'],
                                    lr=cfg.settings['ae_lr'],
                                    n_epochs=cfg.settings['ae_n_epochs'],
                                    batch_size=cfg.settings['ae_batch_size'],
                                    weight_decay=cfg.settings['ae_weight_decay'],
                                    device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)

            ## Train model on datasets
            deepOCC.train(dataset,
                            optimizer_name=cfg.settings['optimizer_name'],
                            lr=cfg.settings['lr'],
                            n_epochs=cfg.settings['n_epochs'],
                            batch_size=cfg.settings['batch_size'],
                            weight_decay=cfg.settings['weight_decay'],
                            device=device,
                            n_jobs_dataloader=n_jobs_dataloader,
                            al_loss=al_loss)

            ''' ============================ '''
            ''' == Test & Pseudo labeling == '''
            ''' ============================ '''
            # print('\n# Test')

            auc, dist, z_data = deepOCC.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

            ascore = dist.copy()

            aucs.append(auc)
            total_dists.append(dist)
            total_z_datas.append(z_data)
            total_z_centers.append(deepOCC.trainer.c.cpu().numpy().copy())

            print('\tTest:: AUC            : {:.4f}\t(Max (Stage {}): {:4f})\t(Stage 0: {:.4f})\n'.format(auc, np.argmax(aucs), max(aucs), aucs[0]))


            ''' ===============================
                == Query for Active Learning ==
                   (random, mlp, db, abs, no)
                =============================== '''
            n_data = dataset.n_data
            numQueries = int(numQueriesRate*n_data)
            ps_numQueries = 0

            training_labels = dataset.train_set.targets

            if dataset_name in ['ionosphere', 'arrhythmia', 'glass']:  # for small datasets
                numQueries = 6

            candidate_idx = np.where(trainset_label_known==0)[0]  # among unlabeled samples
            
            if qs_type != 'no':

                if qs_type == 'random':  # Random
                    sortIdx = np.random.choice(candidate_idx, numQueries)

                if qs_type == 'mlp':  # most-likely positive (high ascore; high confidence)
                    candidate_value = ascore[candidate_idx]
                    sortIdx = np.argsort(candidate_value)[::-1]
                    sortIdx = candidate_idx[sortIdx]

                elif qs_type == 'db':  # decision boundary
                    assert 'soft' in al_loss, 'Invalid One-class Classification'
                    candidate_value = abs(dist - deepOCC.R)
                    candidate_value = candidate_value[candidate_idx]
                    sortIdx = np.argsort(candidate_value)
                    sortIdx = candidate_idx[sortIdx]

                elif qs_type == 'abs':  # adaptive boundary search
                    if al_iter_idx == 0:
                        qp = 0.8  # query point for adaptive linear
                        qps = [qp]
                        sortIdx = np.argsort(ascore)
                        sortIdx = sortIdx[int(n_data*qp):int(n_data*qp)+numQueries]
                    else:
                        n_qanomal = training_labels[queryIdx].sum().item()  # from the previous query information
                        n_qnormal = queryIdx.shape[0] - n_qanomal

                        p_qnormal = n_qnormal/(n_qnormal+n_qanomal)

                        if qp == 1:  # qps[-1] == 1 (if the previous qp is 1)
                            qp = (qp + qps[-2])/2
                        else:
                            qp = 2*(1-qp)*p_qnormal + (2*qp-1)
                        qps.append(qp)

                        sortIdx = np.argsort(ascore[candidate_idx])

                        if int(n_data*qp)+numQueries > sortIdx.shape[0]:  #n_data:
                            # print('reach Max ascore')
                            sortIdx = sortIdx[-numQueries:]
                        else:
                            sortIdx = sortIdx[int(n_data*qp):int(n_data*qp)+numQueries]
                        sortIdx = candidate_idx[sortIdx]
                    total_abs_boundary.append(qp)

                ### COMMON
                candidate = sortIdx[:numQueries]  # top-k
                if len(candidate) > 0:
                    queryIdx = candidate
                    trainset_label_known[queryIdx] = 1
                elif len(candidate)==0:
                    print('No more Esempi to query (All Esempi have been queried)')
                    queryIdx = []

                total_label_known.append(trainset_label_known.copy())

                if ps_abnormal: 
                    trainset_ps_label_known = np.zeros(trainset_ps_label_known.shape)  

                    candidate_idx = np.where(trainset_label_known==0)[0]  

                    labeled_idx = np.where(trainset_label_known==1)[0]

                    labeled_abnormal_idx = np.where(np.array(training_labels)[np.where(trainset_label_known==1)[0]]==1)[0]
                    if len(labeled_abnormal_idx) == 0:
                        candidate_idx = candidate_idx[ascore[candidate_idx] >= ascore[labeled_idx].max()]
                        candidate_value_ascore = ascore[candidate_idx]
                        sortIdx = np.argsort(candidate_value_ascore) 
                        ps_numQueries = numQueries 
                        sortIdx = candidate_idx[sortIdx[-ps_numQueries:]]
                    else:
                        labeled_abnormal_idx = labeled_idx[labeled_abnormal_idx]
                        candidate_idx = candidate_idx[ascore[candidate_idx] >= np.median(ascore[labeled_abnormal_idx])]

                        n_half_of_candidates = int(np.where(trainset_label_known==0)[0].shape[0]*0.5)
                        if candidate_idx.shape[0] > n_half_of_candidates:
                            sortIdx = np.random.choice(candidate_idx, n_half_of_candidates)
                        else:
                            sortIdx = candidate_idx.copy()

                    trainset_ps_label_known[sortIdx] = 1

                    total_pseudo_abnormal.append(trainset_ps_label_known.copy())

            n_known_normal = np.where(np.array(training_labels)[np.where(trainset_label_known==1)[0]]==0)[0].size
            n_known_outlier = np.where(np.array(training_labels)[np.where(trainset_label_known==1)[0]]==1)[0].size
            n_known_normal_new = np.where(np.array(training_labels)[queryIdx]==0)[0].size
            n_known_outlier_new = np.where(np.array(training_labels)[queryIdx]==1)[0].size
            print('\tAL step {} DONE. Query target --> [Normal: {}/ Abnormal: {} + {}]'.format(al_iter_idx,
                                                                        n_known_normal, n_known_outlier, int(sum(trainset_ps_label_known))))
            print('\t                           (new) [Normal: {}/ Abnormal: {}]'.format(n_known_normal_new, n_known_outlier_new))
            
            dataset.update_label_known(trainset_label_known, queryIdx)
            dataset.update_ps_label_known(trainset_ps_label_known)
            print('\tAL Stage time: {:.3f}s'.format(time.time() - t_start_al_iter))

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

        with open('{}/results_r{}_a{}.pickle'.format(result_path, n_repeat_idx, n_al_iter), 'wb') as f:  
            pickle.dump(results, f)
        print('Results saved. :: {}/results_r{}_a{}.pickle\n'.format(result_path, n_repeat_idx, n_al_iter))


    total_aucs = []
    for n_repeat_idx in range(n_repeat):
        with open('{}/results_r{}_a{}.pickle'.format(result_path, n_repeat_idx, n_al_iter), 'rb') as f:  
            res = pickle.load(f)        
        total_aucs.append(res['aucs'])
    total_aucs = np.array(total_aucs)
    
    print('\n')
    print('        Dataset: {}'.format(dataset_name))
    print('           Loss: {}'.format(al_loss))
    print('          Query: {}'.format(qs_type))
    print('Pseudo-abnormal: {}'.format(bool(ps_abnormal)), '\n')

    print('AUC (Average/Std):')
    print([float('{:.2f}'.format(x)) for x in np.mean(total_aucs, axis=0)*100])
    print([float('{:.1f}'.format(x)) for x in np.std(total_aucs, axis=0)*100])

if __name__ == '__main__':
    main()
