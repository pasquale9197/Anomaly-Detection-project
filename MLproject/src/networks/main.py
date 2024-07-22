from .mlp import MLP, MLP_Autoencoder, MLP_UAI

h_dims = {'arrhythmia_mlp': [128, 64],
          'cardio_mlp': [32, 16],
          'ionosphere_mlp':[32, 16],
          'mnist_tab_mlp':[64, 32],
          'glass_mlp':[32, 16],
          'optdigits_mlp':[32, 16],
          'nslkdd_mlp':[32, 16]
          }
rep_dim = {'arrhythmia_mlp': 32,
          'cardio_mlp': 8,
          'ionosphere_mlp':8,
          'mnist_tab_mlp':16,
          'glass_mlp':8,
          'optdigits_mlp':8,
          'nslkdd_mlp':8
          }


def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('arrhythmia_mlp', 'cardio_mlp', 'nslkdd_mlp',
                            'ionosphere_mlp', 'mnist_tab_mlp', 'glass_mlp', 'optdigits_mlp',
                            'ionosphere_mlp_uai', 'arrhythmia_mlp_uai', 'cardio_mlp_uai', 'mnist_tab_mlp_uai', 
                            'glass_mlp_uai', 'optdigits_mlp_uai', 'nslkdd_mlp_uai')
    assert net_name in implemented_networks

    # if net_name in h_dims:
        # print('{} Architecture h_dims: {}'.format(net_name, h_dims[net_name]))
        # print('{} Architecture feat_dims: {}'.format(net_name, rep_dim[net_name]))

    net = None


    if net_name == 'ionosphere_mlp_uai':
        net = MLP_UAI(x_dim=33, h_dims=h_dims['ionosphere_mlp'], rep_dim=rep_dim['ionosphere_mlp'], bias=False)

    if net_name == 'arrhythmia_mlp_uai':
        net = MLP_UAI(x_dim=274, h_dims=h_dims['arrhythmia_mlp'], rep_dim=rep_dim['arrhythmia_mlp'], bias=False)

    if net_name == 'cardio_mlp_uai':
        net = MLP_UAI(x_dim=21, h_dims=h_dims['cardio_mlp'], rep_dim=rep_dim['cardio_mlp'], bias=False)

    if net_name == 'mnist_tab_mlp_uai':
        net = MLP_UAI(x_dim=100, h_dims=h_dims['mnist_tab_mlp'], rep_dim=rep_dim['mnist_tab_mlp'], bias=False)

    if net_name == 'glass_mlp_uai':
        net = MLP_UAI(x_dim=9, h_dims=h_dims['glass_mlp'], rep_dim=rep_dim['glass_mlp'], bias=False)

    if net_name == 'optdigits_mlp_uai':
        net = MLP_UAI(x_dim=64, h_dims=h_dims['optdigits_mlp'], rep_dim=rep_dim['optdigits_mlp'], bias=False)

    if net_name == 'nslkdd_mlp_uai':
        net = MLP_UAI(x_dim=41, h_dims=h_dims['nslkdd_mlp'], rep_dim=rep_dim['nslkdd_mlp'], bias=False)


    if net_name == 'arrhythmia_mlp':
        net = MLP(x_dim=274, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'cardio_mlp':
        net = MLP(x_dim=21, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'nslkdd_mlp':
        net = MLP(x_dim=41, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'ionosphere_mlp':
        net = MLP(x_dim=33, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'mnist_tab_mlp':
        net = MLP(x_dim=100, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'glass_mlp':
        net = MLP(x_dim=9, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'optdigits_mlp':
        net = MLP(x_dim=64, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    # if net_name in h_dims:
        # print('{} Architecture h_dims: {}'.format(net_name, h_dims[net_name]))

    implemented_networks = ('arrhythmia_mlp', 'cardio_mlp', 
                            'ionosphere_mlp', 'mnist_tab_mlp', 'nslkdd_mlp',
                            'glass_mlp', 'optdigits_mlp')

    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'arrhythmia_mlp':
        ae_net = MLP_Autoencoder(x_dim=274, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'cardio_mlp':
        ae_net = MLP_Autoencoder(x_dim=21, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'ionosphere_mlp':
        ae_net = MLP_Autoencoder(x_dim=33, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'nslkdd_mlp':
        ae_net = MLP_Autoencoder(x_dim=41, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'mnist_tab_mlp':
        ae_net = MLP_Autoencoder(x_dim=100, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'glass_mlp':
        ae_net = MLP_Autoencoder(x_dim=9, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    if net_name == 'optdigits_mlp':
        ae_net = MLP_Autoencoder(x_dim=64, h_dims=h_dims[net_name], rep_dim=rep_dim[net_name], bias=False)

    return ae_net
