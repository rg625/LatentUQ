import torch
import torch.nn as nn
import torch.nn.functional as F
from .nnBasis import nnBasis

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class OrthogonalBasis(nn.Module):
    def __init__(self, gmm_dim, num_expansions, nu=0.5, device='cuda'):
        super().__init__()
        self.gen_feature = 128
        self.gmm_dim = gmm_dim
        self.device = device
        self.num_expansion = num_expansions
        self.orthogonal_functions = nnBasis(self.num_expansion).to(device)
        # self.length_scale = nn.Sequential(
        #     nn.Linear(gmm_dim, self.gen_feature),
        #     nn.ReLU(),
        #     nn.Linear(self.gen_feature, self.gen_feature),
        #     nn.ReLU(),
        #     nn.Linear(self.gen_feature, 1),
        #     nn.ReLU()
        # )

        # self.log_sigma = nn.Sequential(
        #     nn.Linear(gmm_dim, self.gen_feature),
        #     nn.ReLU(),
        #     nn.Linear(self.gen_feature, self.gen_feature),
        #     nn.ReLU(),
        #     nn.Linear(self.gen_feature, 1)
        # )

        self.gen = nn.Sequential(
            nn.Linear(gmm_dim, self.gen_feature),
            nn.ReLU(),
            nn.Linear(self.gen_feature, self.gen_feature),
            nn.ReLU(),
            nn.Linear(self.gen_feature, self.num_expansion)
        ).to(device)
        self.nu = torch.tensor(nu, device=device)
        self.eps = torch.tensor(1e-4, device=device)
        self.to(device)

        # Precompute constants
        self.two = torch.tensor(2.0, device=self.device)
        self.pi = torch.tensor(torch.pi, device=self.device)
    
    def mean(self, x, t):
        return torch.zeros_like(t).to(self.device)
    
    def calculate_gamma_scaling(self, log_sigma, ell, d):
        return (
            2 * log_sigma
            + d * torch.log(self.two)
            + (d / 2) * torch.log(self.pi)
            + torch.special.gammaln(self.nu + d / 2)
            - torch.special.gammaln(self.nu)
            + d * torch.log(ell)
        )

    def generate_cosines(self, t, num_samples):
        orthogonal_functions, smoothness = self.orthogonal_functions(t)
        return orthogonal_functions.unsqueeze(0).expand(num_samples, -1, -1), smoothness

    def compute_spectral_weights(self, gamma_scaling, ell, j, d):
        log_spectral_weights = (-self.nu - d / 2) * torch.log(ell**2 * j**2 + 1)
        spectral_weights = torch.exp(0.5 * (gamma_scaling + log_spectral_weights))
        return spectral_weights

    def sample_function(self,
                        t,
                        x,
                        num_samples = 20,
                        d=1,
                        ):
        # log_sigma = self.log_sigma(x)
        # ell = (self.length_scale(x)+ self.eps)
        # j = torch.pi * torch.arange(1, self.num_expansion + 1, device=device).float()

        # gamma_scaling = self.calculate_gamma_scaling(log_sigma, ell, d)
        cosines, smoothness = self.generate_cosines(t, num_samples)
        cosines = cosines.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        # spectral_weights = self.compute_spectral_weights(gamma_scaling, ell, j, d).unsqueeze(1).expand(-1, num_samples, -1)
        spectral_weights = self.gen(x).unsqueeze(1).expand(-1, num_samples, -1)
        # spectral_weights_max = torch.max(spectral_weights, dim=1, keepdim=True)[0]
        # spectral_weights = spectral_weights / spectral_weights_max
        random_weights_cos = torch.randn_like(spectral_weights, device=device)

        # return F.softplus(torch.einsum('bSJ, bStJ -> Sbt', spectral_weights * random_weights_cos, cosines)).squeeze()
        return torch.einsum('bSJ, bStJ -> Sbt', spectral_weights * random_weights_cos, cosines).squeeze(), smoothness

    def forward(self, x, time, num_samples = 1):
        mu =  self.mean(x, time).squeeze()
        random_function = self.sample_function(time, x, num_samples)[0]
        return mu + random_function