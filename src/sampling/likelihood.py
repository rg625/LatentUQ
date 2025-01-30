# src/sampling/likelihood.py

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian_likelihood(x, x_hat, sigma=torch.tensor(1.)):
    """
    Compute Gaussian likelihood between true and predicted data.
    """
    x = x.to(device)
    x_hat = x_hat.reshape(x.shape).to(device)
    sigma_sq = sigma ** 2
    diff = (x - x_hat).unsqueeze(-1)
    return -torch.einsum('...ij,...ij->...j', diff, diff).squeeze() / (2. * sigma_sq)