import torch
import torch.nn as nn
from torch.distributions import Normal

class GaussianSoftMaxCriterion(nn.Module):
    def __init__(self, num_classes, mu=0, sigma=1, lambda_par=1):
        super(GaussianSoftMaxCriterion, self).__init__()
        self.mu = nn.Parameter(torch.ones(num_classes) * mu)
        self.sigma = nn.Parameter(torch.ones(num_classes) * sigma)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.lambda_par = lambda_par

    def forward(self, X, y):
        Pr = torch.zeros_like(X)
        for i in range(Pr.size(1)):
            Pr[:, i] = Normal(self.mu[i], self.sigma[i]).cdf(X[:, i].double())
        Pr = (Pr.type_as(X)) * self.lambda_par + X
        
        return self.ce_criterion(Pr, y)

    def predict(self, X):
        Pr = torch.zeros_like(X)
        for i in range(Pr.size(1)):
            Pr[:, i] = Normal(self.mu[i], self.sigma[i]).cdf(X[:, i].double())
        Pr = (Pr.type_as(X)) * self.lambda_par + X
        exp_sum = torch.exp(Pr).sum(dim=1, keepdim=True)
        pred = torch.exp(Pr) / exp_sum
        
        return pred
    

if __name__ == '__main__':
    model = GaussianSoftMaxCriterion(20)
    print(dict(model.named_parameters()))
