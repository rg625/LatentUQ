# src/sampling/posterior.py

import torch
import numpy as np
from sampling.likelihood import gaussian_likelihood
from sampling.prior import log_prior
from utils.helpers import check_nans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_post(x, z, means, lower_cholesky, weights, time, model, pushforward, log_likelihood_sigma, testing=False):
    """
    Compute the logarithm of the posterior distribution.
    """
    if testing:
        num_samples = 1
    else:
        num_samples = 300

    a = model.sample_function(time, z, num_samples=num_samples)
    x = x.unsqueeze(0).expand(num_samples, -1, -1)
    log_p_y_given_z = gaussian_likelihood(x.squeeze(), pushforward(a), log_likelihood_sigma)
    return torch.logsumexp(log_p_y_given_z, dim=0) + log_prior(z, means, lower_cholesky, weights)

def grad_log(z, log_dist, max_value=1e3):
    """
    Compute the gradient of the logarithm of a distribution with respect to the latent variables.
    """
    grad = torch.autograd.grad(log_dist, z, grad_outputs=torch.ones_like(log_dist), create_graph=True)[0]
    return torch.clamp(grad, min=-max_value, max=max_value)

def update_z(z, step_size, grad_log_distribution):
    """
    Update the latent variables using the Langevin update rule.
    """
    z = z.detach().clone().requires_grad_()
    grad_log_distribution = grad_log_distribution.detach().clone().requires_grad_()
    normal_sample = torch.randn(z.size(), device=device)
    z = z + step_size**2 * grad_log_distribution + step_size * normal_sample * torch.sqrt(torch.tensor(2.)).to(device)
    return z

def q_mala(z, z_proposed, grad_log, step_size):
    """
    Compute the logarithm of the MALA (Metropolis-Adjusted Langevin Algorithm) acceptance probability.
    """
    diff = z - z_proposed - step_size**2 * grad_log
    return -torch.norm(diff.view(z.shape[0], -1), dim=-1) ** 2 / (4 * step_size**2)

def langevin(x, z, means, lower_cholesky, weights, time, step_size, num_steps, model, pushforward, log_likelihood_sigma, plot=False):
    """
    Perform Langevin dynamics for sampling latent variables.
    """
    z.requires_grad_(True)
    acceptance_count = torch.zeros_like(torch.empty(z.shape[0], 1)).to(device)
    acceptance_rate_evol = []
    grad_energy_evol = []

    check_sampler('ula')

    if plot:
        ULA_chain_z = torch.zeros((num_steps+1, *z.size()))
        ULA_chain_x = torch.zeros((num_steps+1, z.size(0), len(time)))
        ULA_chain_z[0] = z.detach()
        ULA_chain_x[0] = model(z, time).detach()

    average = []
    for i in range(num_steps):
        log_dist = log_post(x, z, means, lower_cholesky, weights, time, model, pushforward, log_likelihood_sigma)
        grad_log_dist = grad_log(z, log_dist)

        grad_energy_evol.append(torch.norm(grad_log_dist, dim=-1).cpu().data.numpy())

        z = update_z(z, step_size, grad_log_dist)
        check_nans(z)
        if num_steps - i < 10:
            average.append(z)

        if plot:
            ULA_chain_z[i+1] = z.detach()
            ULA_chain_x[i+1] = model(z, time).detach()

    acceptance_rate_evol = np.array(acceptance_rate_evol) / num_steps
    grad_energy_evol = np.array(grad_energy_evol)
    x_chain = ULA_chain_x.cpu().data.numpy()
    z_chain = ULA_chain_z.cpu().data.numpy()
    z = torch.stack(average, dim=0).mean(dim=0)

    return z.detach(), np.array(x_chain), np.array(z_chain), grad_energy_evol, acceptance_rate_evol