import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_names = ['ionosphere', 'arrhythmia', 'cardio', 'mnist_tab', 'glass', 'optdigits']
al_losses = ['soft_boundary', 'soft_boundary_nce', 'one_class', 'nce', 'dsad', 'one_class_uai']
qs_types = ['random', 'mlp', 'db', 'abs']

df = pd.DataFrame(columns = ['dataset_name', 'al_loss', 'qs_type', 'ps_abnorm']+['stage{}m'.format(x) for x in range(6)]+['stage{}s'.format(x) for x in range(6)])

for dataset_name in dataset_names:
    for al_loss in al_losses:
        for qs_type in qs_types:
            if 'soft' not in al_loss and qs_type == 'db':
                continue
            
            if 'uai' in al_loss and qs_type != 'mlp':
                continue
            
            ps_abnormal = 0
            if qs_type == 'db' or qs_type == 'abs':
                ps_abnormal = 1
            
            with open('log-{}-{}-{}-ps{}.txt'.format(dataset_name, al_loss, qs_type, ps_abnormal)) as f:
                lines = f.readlines()

            mean = [*map(float, lines[-2][11:-2].split(','))]
            std = [*map(float, lines[-1][11:-2].split(','))]

            res = {'dataset_name' : dataset_name, 'al_loss' : al_loss, 'qs_type' : qs_type, 'ps_abnorm' : ps_abnormal}
            res.update({'stage{}m'.format(x) : mean[x] for x in range(6)})
            res.update({'stage{}s'.format(x) : std[x] for x in range(6)})

            # append rows to an empty DataFrame
            # df = df.append(res, ignore_index = True)
            df = pd.concat([df, pd.DataFrame.from_records([res])])

# Bar graph:
bars = False
bars = True

perData = False
# perData = True

if bars:
    for al_loss in ['one_class', 'dsad', 'nce']:
        dff = df[df['al_loss']==al_loss][['qs_type', 'stage0m', 'stage1m', 'stage2m', 'stage3m', 'stage4m', 'stage5m']].groupby('qs_type').mean().T
        ax = dff[['random', 'mlp', 'abs']].plot.bar(rot=0); plt.ylim([50, 100]); plt.title(al_loss)
        # plt.show()

    for qs_type in ['random', 'mlp', 'abs']:
        dff = df[df['qs_type']==qs_type][['al_loss', 'stage0m', 'stage1m', 'stage2m', 'stage3m', 'stage4m', 'stage5m']].groupby('al_loss').mean().T
        ax = dff[['one_class', 'dsad', 'nce']].plot.bar(rot=0); plt.ylim([50, 100]); plt.title(qs_type)
        # plt.show()

    for al_loss in ['soft_boundary', 'soft_boundary_nce']:
        dff = df[df['al_loss']==al_loss][['qs_type', 'stage0m', 'stage1m', 'stage2m', 'stage3m', 'stage4m', 'stage5m']].groupby('qs_type').mean().T
        ax = dff[['random', 'mlp', 'db', 'abs']].plot.bar(rot=0); plt.ylim([50, 100]); plt.title(al_loss)
        # plt.show()    

    for qs_type in ['random', 'mlp', 'db', 'abs']:
        dff = df[df['qs_type']==qs_type][['al_loss', 'stage0m', 'stage1m', 'stage2m', 'stage3m', 'stage4m', 'stage5m']].groupby('al_loss').mean().T
        ax = dff[['soft_boundary', 'soft_boundary_nce']].plot.bar(rot=0); plt.ylim([50, 100]); plt.title(qs_type)
        # plt.show()
    plt.show()

if perData:
    for dataset_name in dataset_names:
        df['tag'] = df['al_loss'] + ' + ' + df['qs_type']
        dff = df[(df['dataset_name']==dataset_name) & ((df['al_loss']=='soft_boundary') | (df['al_loss']=='soft_boundary_nce'))]
        dff = dff[dff['qs_type']!='random']
        plt.figure(); plt.plot(dff.values[:, 4:10].T, label=dff['tag']); plt.legend(); plt.title(dataset_name)

        dff = df[(df['dataset_name']==dataset_name) & ((df['al_loss']=='one_class') | (df['al_loss']=='dsad') | (df['al_loss']=='nce') | (df['al_loss'] == 'one_class_uai'))]
        dff = dff[dff['qs_type']!='random']
        plt.figure(); plt.plot(dff.values[:, 4:10].T, label=dff['tag']); plt.legend(); plt.title(dataset_name)

        plt.show()

pdb.set_trace()
