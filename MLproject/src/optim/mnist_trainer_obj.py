import json
import torch

from MLProject.MLproject.src.datasets.mnist import MNISTADDDataset
from MLProject.MLproject.src.networks.Mnist_main import build_autoencoder
from MLProject.MLproject.src.networks.SVDNetMnist import MLP_MNIST
from MLProject.MLproject.src.optim.AEMnist_trainer import AEMnist_trainer
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
        self.batch_size = 30
        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

        self.model_pretrained = False

    def set_network(self, net_name: str):
        self.net_name = net_name
        self.net = MLP_MNIST(x_dim=784, h_dims=[512, 256], rep_dim=64, bias=False, num_classes=10)

    # pensa se fare pretrain
    def pretrain(self, dataset: MNISTADDDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
                 batch_size: int = 30, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""
        self.ae_net = build_autoencoder()

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AEMnist_trainer(optimizer_name, lr=lr, n_epochs=n_epochs,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time
        self.ae_results['test_time'] = self.ae_trainer.test_time
        self.ae_results['test_auc'] = self.ae_trainer.test_auc

        # Initialize Deep SAD network weights from pre-trained encoder
        # self.init_network_weights_from_pretraining()
        self.model_pretrained = True

    def train(self, dataset: MNISTADDDataset, optimizer_name: str = 'adam', lr: float = 0.001,
              n_epochs: int = 50, batch_size: int = 30, weight_decay: float = 1e-6, device: str = 'cuda',
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

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k[8:]: v for k, v in ae_net_dict.items() if (k[8:] in net_dict) and ('encoder' in k)}

        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)

        # Load the new state_dict
        self.net.load_state_dict(net_dict)


