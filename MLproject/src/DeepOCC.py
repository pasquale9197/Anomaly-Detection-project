import pdb
import json
import torch

from datasets.base_dataset import BaseADDataset

from networks.main import build_network, build_autoencoder

from optim.ae_trainer import AETrainer
from optim.DeepOCC_trainer import DeepOCCTrainer
from optim.DeepOCC_UAI_trainer import DeepOCCUAITrainer
 

class DeepOCC(object):
    """A class for the Deep SVDD / SAD method.

    Attributes:
        nu: soft-boundary Deep SVDD hyperparameter (must be 0 < nu <= 1)
        R: soft-boundary Deep SVDD hyperparameter
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self):

        self.nu = 0.5  # fixed  "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.R = 0.0

        self.eta = 1  # fixed  "For hyperparameter eta, it must hold: 0 < eta."
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

        self.model_pretrained = False

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        if 'uai' in self.net_name:
            self.ae_net = build_autoencoder(self.net_name[:-4])
        else:
            self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time
        self.ae_results['test_time'] = self.ae_trainer.test_time
        self.ae_results['test_auc'] = self.ae_trainer.test_auc

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()
        self.model_pretrained = True

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, al_mode=None, al_loss=-1):
        """Trains the Deep OCC model on the training Esempi."""
        self.al_mode = al_mode
        self.al_loss = al_loss

        self.optimizer_name = optimizer_name

        if 'uai' in self.al_loss:
            self.trainer = DeepOCCUAITrainer(self.R, self.c, self.nu, self.eta,
                                      n_epochs=n_epochs, batch_size=batch_size, weight_decay=weight_decay,
                                      lr=lr, optimizer_name=optimizer_name,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader, al_loss=al_loss)
        else:
            self.trainer = DeepOCCTrainer(self.R, self.c, self.nu, self.eta,
                                      n_epochs=n_epochs, batch_size=batch_size, weight_decay=weight_decay,
                                      lr=lr, optimizer_name=optimizer_name,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader, al_loss=al_loss)

        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list
        self.results['td_ascores'] = self.trainer.td_ascores
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test Esempi."""

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

        return self.trainer.test_auc, self.trainer.scores, self.trainer.outputs

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

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)
