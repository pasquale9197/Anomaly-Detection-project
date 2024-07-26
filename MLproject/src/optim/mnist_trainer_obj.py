import json
import torch

from MLProject.MLproject.src.datasets.mnist import MNISTADDDataset
from MLProject.MLproject.src.networks.SVDNetMnist import MLP_MNIST
from MLProject.MLproject.src.optim.mnist_trainer import DeepSVDMnistTrainer


class Mnist_trainer_obj(object):
    def __init__(self):
        self.nu = 0.5
        self.R = 0.0

        self.eta = 1
        self.c = None

        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None
        self.net = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None
        }

        self.model_pretrained = False

    def set_network(self, net_name: str):
        self.net_name = net_name
        self.net = MLP_MNIST(x_dim=784, h_dims=[64,128], rep_dim=32, bias=False, num_classes=10)

    # pensa se fare pretrain

    def train(self, dataset: MNISTADDDataset, optimizer_name: str = 'adam', lr: float = 0.001,
              n_epochs: int = 50, batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cpu',
              n_jobs_dataloader: int = 0, al_mode=None, al_loss=-1):
        self.al_mode = al_mode
        self.al_loss = al_loss

        self.optimizer_name = optimizer_name

        self.trainer = DeepSVDMnistTrainer(self.R, self.c, self.nu, self.eta,
                                           n_epochs=n_epochs, batch_size=batch_size, weight_decay=weight_decay,
                                           lr=lr, optimizer_name=optimizer_name, device=device,
                                           n_jobs_dataloader=n_jobs_dataloader, al_loss=al_loss)

        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())
        self.c = self.trainer.c.cpu().data.numpy().tolist()
        self.results['train_time'] = self.trainer.train_time
        self.results['td_ascores'] = self.trainer.td_ascores

    def test(self, dataset: MNISTADDDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

        return self.trainer.test_auc, self.trainer.scores, self.trainer.outputs


    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None
        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict
                    }, export_model)

