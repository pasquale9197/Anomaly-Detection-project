import torch
import torch.nn as nn
import torch.nn.functional as F
from MLProject.MLproject.src.base import BaseNet
import sys
import os



class MLP_MNIST(BaseNet):

    def __init__(self, x_dim=784, h_dims=[64, 128], rep_dim=32, bias=False, num_classes=10):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i-1], neurons[i], bias=bias)
                  for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

        self.classifier = nn.Linear(rep_dim, num_classes)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        rep = self.code(x)
        return rep, self.classifier(rep)

    def forward_uai(self, z, s):
        return self.output_activation(self.reg(torch.cat([z, torch.unsqueeze(s, 1)], dim=1)))


class Linear_BN_leakyReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))

