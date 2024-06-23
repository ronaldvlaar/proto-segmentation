"""
G-softmax from https://github.com/luoyan407/gsoftmax
"""

import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np


def compactness(sigmas): return 1/sigmas

def separability(mu, sigma, num_classes):
    class_separability = np.zeros(num_classes)
    
    for i in range(num_classes):
        kld_sum = 0
        for j in range(num_classes):
            if i != j:
                sigma_i = sigma[i]
                sigma_j = sigma[j]
                mu_i = mu[i]
                mu_j = mu[j]
                
                kld = np.log(sigma_j / sigma_i) + (sigma_i**2 + (mu_i - mu_j)**2) / (2 * sigma_j**2) - 0.5
                kld2 = np.log(sigma_i / sigma_j) + (sigma_j**2 + (mu_j - mu_i)**2) / (2 * sigma_i**2) - 0.5
                kld_sum += (kld+kld2)
        
        class_separability[i] = kld_sum / (2*(num_classes-1))

    return class_separability


class GaussianSoftMaxCriterion(nn.Module):
    def __init__(self, num_classes, mu=-0.05, sigma=1.0, scale=1.0):
        super(GaussianSoftMaxCriterion, self).__init__()
        self.num_classes = num_classes
        self.mu = nn.Parameter(torch.ones(num_classes) * mu)
        self.sigma = nn.Parameter(torch.ones(num_classes) * sigma)
        self.cecriterion = nn.CrossEntropyLoss()
        self.scale = scale

    def forward(self, _input, _target):
        input = _input.clone()
        normal_dist = Normal(self.mu.to(input.device), torch.clamp(self.sigma.to(input.device), min=1e-6))
        prob = normal_dist.cdf(input)
        prob = prob * self.scale + input
        
        self.output = self.cecriterion(prob, _target)
        
        return self.output

    def predict(self, _input):
        input = _input.clone()
        normal_dist = Normal(self.mu.to(input.device), torch.clamp(self.sigma.to(input.device), min=1e-6))
        prob = normal_dist.cdf(input)
        prob = prob * self.scale + input
        
        output = F.softmax(prob, dim=1)

        return output
    
    def get_compactness(self):
        return compactness(self.sigma.cpu().detach().numpy())

    def get_separability(self):
        return separability(self.mu.cpu().detach().numpy(), self.sigma.cpu().detach().numpy(), self.num_classes)
    
    def get_mu(self):
        return self.mu.cpu().detach().numpy()
    
    def get_sigma(self):
        return self.sigma.cpu().detach().numpy()

    def __str__(self):
        return f'{self.__class__.__name__}(mu={self.mu}, sigma={self.sigma}, scale={self.scale})'
