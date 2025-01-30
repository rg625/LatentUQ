import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_gamma_scaling(log_sigma, ell, d, nu = 0.5, eps = 1e-4):
    return (
        2 * log_sigma
        + d * torch.log(torch.tensor(2.0, device=device))
        + (d / 2) * torch.log(torch.tensor(torch.pi, device=device))
        + torch.special.gammaln(nu + d / 2)
        - torch.special.gammaln(nu)
        + d * torch.log(ell)
    )

def generate_cosines(t, batch_size, num_samples, num_expansion = 30):
    j = torch.pi * torch.arange(1, num_expansion + 1, device=device).float()
    time_term = j * t.unsqueeze(-1)
    time_term = time_term.unsqueeze(0).expand(batch_size, -1, -1)
    cosines = torch.cos(time_term).unsqueeze(1).expand(-1, num_samples, -1, -1)
    return cosines

def compute_spectral_weights(gamma_scaling, ell, j, d, nu = 0.5):
    log_spectral_weights = (-nu - d / 2) * torch.log(ell**2 * j**2 + 1)
    spectral_weights = torch.exp(0.5 * (gamma_scaling + log_spectral_weights))
    return spectral_weights

def sample_true_function(
    t, 
    batch_size, 
    log_sigma=torch.log(torch.tensor(2.)), 
    ell=torch.tensor(0.5), 
    nu=torch.tensor(2.5), 
    n_exp=30, 
    eps=1e-4, 
    d=1,
    num_samples = 1):
    log_sigma = log_sigma*torch.ones(5000, 1, device=device)
    ell = ell*torch.ones(5000, 1, device=device)
    nu = nu*torch.ones(5000, 1, device=device)
    j = torch.pi * torch.arange(1, n_exp + 1, device=device).float()

    gamma_scaling = calculate_gamma_scaling(log_sigma, ell, d, nu, eps)
    cosines = generate_cosines(t, batch_size, num_samples, num_expansion = n_exp)
    spectral_weights = compute_spectral_weights(gamma_scaling, ell, j, d, nu).unsqueeze(1).expand(-1, num_samples, -1)
    random_weights_cos = torch.randn_like(spectral_weights, device=device)

    return F.softplus(torch.einsum('bSJ, bStJ -> Sbt', spectral_weights * random_weights_cos, cosines)).squeeze()