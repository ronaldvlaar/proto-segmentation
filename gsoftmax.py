import torch
import torch.nn as nn
from torch.distributions import Normal

class GaussianSoftMaxCriterion(nn.Module):
    def __init__(self, num_classes, mu=0, sigma=1, lambda_par=1):
        super(GaussianSoftMaxCriterion, self).__init__()
        self.mu = nn.Parameter(torch.ones(num_classes) * mu)
        self.sigma = nn.Parameter(torch.ones(num_classes) * sigma)
        self.num_classes = num_classes
        self.ce_criterion = nn.CrossEntropyLoss()
        self.lambda_par = lambda_par
    
    def forward(self, X, y):
        mu = self.mu.unsqueeze(0).expand(X.size(0), self.num_classes)
        sigma = torch.clamp(self.sigma.unsqueeze(0).expand(X.size(0), self.num_classes), min=1e-6)
        
        Pr = Normal(mu, sigma).cdf(X.double())
        Pr = Pr.type_as(X) * self.lambda_par + X

        return self.ce_criterion(Pr, y)

    # def forward(self, X, y):
    #     Pr = torch.zeros_like(X)

    #     for i in range(self.num_classes):
    #         Pr[:, i] = Normal(self.mu[i], self.sigma[i]).cdf(X[:, i].double())
    #     Pr = (Pr.type_as(X)) * self.lambda_par + X

    #     return self.ce_criterion(Pr, y)
    
    def predict(self, X):
        # print(X, len(X))
        # Pr = torch.zeros_like(X)
        # for i in range(self.num_classes):
        #     Pr[:, i] = Normal(self.mu[i], self.sigma[i]).cdf(X[:, i].double())
        # Pr = (Pr.type_as(X)) * self.lambda_par + X
        # exp_sum = torch.exp(Pr).sum(dim=1, keepdim=True)
        # pred = torch.exp(Pr) / exp_sum

        mu = self.mu.unsqueeze(0).expand(X.size(0), self.num_classes)
        sigma = torch.clamp(self.sigma.unsqueeze(0).expand(X.size(0), self.num_classes), min=1e-10)
        Pr = Normal(mu, sigma).cdf(X.double())
        Pr = Pr * self.lambda_par + X
        exp_sum = torch.exp(Pr).sum(dim=1, keepdim=True)
        pred = torch.exp(Pr) / exp_sum

        return pred
    
    def get_compactness(self):
        return 1/self.sigma.cpu().detach().numpy()

    def get_separability(self):
        class_separability = torch.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            kld_sum = 0
            for j in range(self.num_classes):
                if i != j:
                    sigma_i = self.sigma[i]
                    sigma_j = self.sigma[j]
                    mu_i = self.mu[i]
                    mu_j = self.mu[j]
                    
                    kld = torch.log(sigma_j / sigma_i) + (sigma_i**2 + (mu_i - mu_j)**2) / (2 * sigma_j**2) - 0.5
                    kld2 = torch.log(sigma_i / sigma_j) + (sigma_j**2 + (mu_j - mu_i)**2) / (2 * sigma_i**2) - 0.5
                    kld_sum += (kld+kld2)
            
            class_separability[i] = kld_sum / (2*(self.num_classes-1))

        return class_separability.detach().numpy()
    

if __name__ == '__main__':
    model = GaussianSoftMaxCriterion(20)
    model.get_separability()
    # print(list(model.parameters()))
    # print(dict(model.named_parameters()))
