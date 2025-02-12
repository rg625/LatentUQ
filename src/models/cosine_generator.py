import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CosineBasis(nn.Module):
    def __init__(self, gmm_dim, nu=0.5, device='cpu'):
        super().__init__()
        self.gen_feature = 32
        self.gmm_dim = gmm_dim
        self.device = device

        self.length_scale = nn.Sequential(
            nn.Linear(gmm_dim, self.gen_feature),
            nn.ReLU(),
            nn.Linear(self.gen_feature, self.gen_feature),
            nn.ReLU(),
            nn.Linear(self.gen_feature, 1),
            nn.ReLU()
        )

        self.log_sigma = nn.Sequential(
            nn.Linear(gmm_dim, self.gen_feature),
            nn.ReLU(),
            nn.Linear(self.gen_feature, self.gen_feature),
            nn.ReLU(),
            nn.Linear(self.gen_feature, 1)
        )

        self.nu = torch.tensor(nu, device=device)
        self.eps = torch.tensor(1e-4, device=device)
        self.num_expansion = 30
        self.to(device)

    def mean(self, x, t):
        return torch.zeros_like(t).to(device)

    def calculate_gamma_scaling(self, log_sigma, ell, d):
        return (
            2 * log_sigma
            + d * torch.log(torch.tensor(2.0, device=device))
            + (d / 2) * torch.log(torch.tensor(torch.pi, device=device))
            + torch.special.gammaln(self.nu + d / 2)
            - torch.special.gammaln(self.nu)
            + d * torch.log(ell)
        )

    def generate_cosines(self, t, batch_size, num_samples, num_expansion = 30):
        j = torch.pi * torch.arange(1, num_expansion + 1, device=device).float()
        time_term = j * t.unsqueeze(-1)
        time_term = time_term.unsqueeze(0).expand(batch_size, -1, -1)
        cosines = torch.cos(time_term).unsqueeze(1).expand(-1, num_samples, -1, -1)
        return cosines

    def compute_spectral_weights(self, gamma_scaling, ell, j, d):
        log_spectral_weights = (-self.nu - d / 2) * torch.log(ell**2 * j**2 + 1)
        spectral_weights = torch.exp(0.5 * (gamma_scaling + log_spectral_weights))
        return spectral_weights

    def sample_function(self,
                        t,
                        x,
                        num_samples = 20,
                        n_exp=30, 
                        d=1,
                        ):
        log_sigma = self.log_sigma(x)
        ell = (self.length_scale(x)+ self.eps)
        batch_size = x.size(0)
        j = torch.pi * torch.arange(1, n_exp + 1, device=device).float()

        gamma_scaling = self.calculate_gamma_scaling(log_sigma, ell, d)
        cosines = self.generate_cosines(t, batch_size, num_samples, num_expansion = n_exp)
        spectral_weights = self.compute_spectral_weights(gamma_scaling, ell, j, d).unsqueeze(1).expand(-1, num_samples, -1)
        random_weights_cos = torch.randn_like(spectral_weights, device=device)

        return F.softplus(torch.einsum('bSJ, bStJ -> Sbt', spectral_weights * random_weights_cos, cosines)).squeeze()
        # return torch.einsum('bSJ, bStJ -> Sbt', spectral_weights * random_weights_cos, cosines).squeeze()

    def forward(self, x, time, num_samples = 1):
        mu =  self.mean(x, time).squeeze()
        random_function = self.sample_function(time, x, num_samples)
        return mu + random_function