from .mlp import MLP, MLP_Autoencoder, MLP_UAI

h_dims = {'mlp': [256, 128]}
rep_dim = {'mlp': 64}


def build_network():
    """Builds the neural network."""
    net = MLP_UAI(x_dim=784, h_dims=h_dims['mlp'], rep_dim=rep_dim['mlp'], bias=False)
    return net


def build_autoencoder():
    """Builds the corresponding autoencoder network."""
    ae_net = MLP_Autoencoder(x_dim=784, h_dims=h_dims['mlp'], rep_dim=rep_dim['mlp'], bias=False)

    return ae_net