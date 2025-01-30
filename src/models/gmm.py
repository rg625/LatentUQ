import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GMM(nn.Module):
    def __init__(self, components, dimensions=2, width = 1., std = 1.):
        super().__init__()

        self.components = components
        self.dimensions = dimensions
        self.width = torch.tensor(width).to(device)
        self.std = torch.tensor(std).to(device)
        self.means = nn.Parameter(self.width*torch.randn((self.components, self.dimensions), device = device), requires_grad=True)
        self.log_weights = nn.Parameter(torch.ones((self.components), device = device), requires_grad=True)
        self.covs = nn.Parameter(self.std*torch.randn((self.components, self.dimensions * (self.dimensions + 1) // 2), device = device), requires_grad=True)


    def create_cov(self):
        lower_cholesky = torch.zeros((self.components, self.dimensions, self.dimensions), device=device)
        for comp in range(self.components):
            rows, cols = torch.tril_indices(row=self.dimensions, col=self.dimensions, offset=0)
            lower_cholesky[comp][rows, cols] = self.covs[comp]

            # Ensure positive diagonal elements
            diag_indices = torch.arange(self.dimensions)
            lower_cholesky[comp][diag_indices, diag_indices] = F.softplus(lower_cholesky[comp][diag_indices, diag_indices])
            # lower_cholesky[comp] = F.softplus(torch.diag(self.covs[comp]))
            lower_cholesky[comp] += 1e-6*torch.eye(self.dimensions).to(device)
        return lower_cholesky.to(device)
        
    def forward(self, batch_size):
        self.weights = F.softmax(self.log_weights, dim=0)
        component_indices = torch.multinomial(self.weights, batch_size, replacement=True)
        sampled_means = self.means[component_indices]
        lower_cholesky = self.create_cov()
        lower_cholesky_indices = lower_cholesky[component_indices]

        random_samples = torch.randn((batch_size, self.dimensions), device = device)
        samples = sampled_means + torch.einsum('bij,bj->bi', lower_cholesky_indices, random_samples)
        # normalized_samples = self.bn(samples)

        return samples, self.means, lower_cholesky, self.weights