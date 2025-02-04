import torch
import numpy as np
from .likelihood import *
from .prior import *
from src.utils.helpers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_likelohood(**kwargs):
    """
    Compute the logarithm of the posterior distribution.
    """
    x = kwargs['x']
    z = kwargs['z']
    time = kwargs['time']
    model = kwargs['model']
    pushforward = kwargs['pushforward']
    log_likelihood_sigma = kwargs['log_likelihood_sigma']
    testing = kwargs.get('testing', False)
    num_samples = kwargs.get('num_samples', 300)

    if testing:
        num_samples = 1

    # print(testing)
    a, smoothness = model.sample_function(time, z, num_samples=num_samples)
    x = x.unsqueeze(0).expand(num_samples, -1, -1)
    log_p_y_given_z = gaussian_likelihood(x = x.squeeze(), x_hat = pushforward(a), sigma = log_likelihood_sigma)
    return torch.logsumexp(log_p_y_given_z, dim=0), smoothness

def log_post(**kwargs):
    """
    Compute the logarithm of the posterior distribution.
    """
    x = kwargs['x']
    z = kwargs['z']
    means = kwargs['means']
    lower_cholesky = kwargs['lower_cholesky']
    weights = kwargs['weights']
    time = kwargs['time']
    model = kwargs['model']
    pushforward = kwargs['pushforward']
    log_likelihood_sigma = kwargs['log_likelihood_sigma']
    testing = kwargs.get('testing', False)

    return log_likelohood(
        x=x,
        z=z,
        time=time,
        model=model,
        pushforward=pushforward,
        log_likelihood_sigma=log_likelihood_sigma,
        testing=testing
    )[0] + log_prior(z = z, means = means, lower_cholesky = lower_cholesky, weights = weights)

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

def langevin(**kwargs):
    """
    Perform Langevin dynamics for sampling latent variables.
    """
    x = kwargs['x']
    z = kwargs['z']
    means = kwargs['means']
    lower_cholesky = kwargs['lower_cholesky']
    weights = kwargs['weights']
    time = kwargs['time']
    step_size = kwargs['step_size']
    num_steps = kwargs['num_steps']
    model = kwargs['model']
    pushforward = kwargs['pushforward']
    log_likelihood_sigma = kwargs['log_likelihood_sigma']
    plot = kwargs.get('plot', False)
    sampler = kwargs.get('sampler', 'ula')
    testing = kwargs.get('testing', False)
    last_samples = 5

    z.requires_grad_(True)
    acceptance_count = torch.zeros_like(torch.empty(z.shape[0], 1)).to(device)
    acceptance_rate_evol = []
    grad_energy_evol = []

    check_sampler(sampler)

    ULA_chain_z = torch.zeros((num_steps+1, *z.size() ))
    ULA_chain_x = torch.zeros((num_steps+1, z.size(0), len(time)))
    if plot:
        ULA_chain_z[0] = z.detach()
        ULA_chain_x[0] = model(z, time).detach()

    average = torch.zeros((num_steps//last_samples, *z.size())).to(device)
    for i in range(num_steps):
        log_dist = log_post(
            x=x,
            z=z,
            means=means,
            lower_cholesky=lower_cholesky,
            weights=weights,
            time=time,
            model=model,
            pushforward=pushforward,
            log_likelihood_sigma=log_likelihood_sigma,
            testing=testing)
        grad_log_dist = grad_log(z, log_dist)

        grad_energy_evol.append(torch.norm(grad_log_dist, dim=-1).cpu().data.numpy())

        if sampler == 'ula':  # If using the Unadjusted Langevin Algorithm (ULA)
            z = update_z(z, step_size, grad_log_dist)
            check_nans(z)
            if num_steps-i<num_steps//last_samples:
                average[i-(num_steps-num_steps//last_samples)] = z
        else:  # If using the Metropolis-Adjusted Langevin Algorithm (MALA)
            z_proposed = update_z(z, step_size, grad_log_dist)
            log_dist_new = log_post(
                x=x,
                z=z_proposed,
                means=means,
                lower_cholesky=lower_cholesky,
                weights=weights,
                time=time,
                model=model,
                pushforward=pushforward,
                log_likelihood_sigma=log_likelihood_sigma,
                testing=testing
            )
            grad_log_dist_new = grad_log(z_proposed, log_dist_new)

            # Calculate the Metropolis-Hastings acceptance probability
            log_alpha = (log_dist_new.view(-1, 1) - log_dist.view(-1, 1) +
                        q_mala(z, z_proposed, grad_log_dist_new, step_size).view(-1, 1) -
                        q_mala(z_proposed, z, grad_log_dist, step_size).view(-1, 1))

            # Generate random values for acceptance test
            log_u = torch.log(torch.rand(log_alpha.size()).to(device))

            # Accept or reject the proposal based on the acceptance probability
            accept = log_u < log_alpha
            acceptance_count += accept
            acceptance_rate_evol.append(acceptance_count.cpu().data.numpy())
    
            # Update the chain with the accepted proposals
            z = torch.where(accept, z_proposed, z)
        if plot:
            ULA_chain_z[i+1] = z.detach()
            ULA_chain_x[i+1] = model(z, time).detach()

    acceptance_rate_evol = np.array(acceptance_rate_evol) / num_steps
    grad_energy_evol = np.array(grad_energy_evol)
    x_chain = ULA_chain_x.cpu().data.numpy()
    z_chain = ULA_chain_z.cpu().data.numpy()
    z = average.mean(dim=0)

    return z.detach(), np.array(x_chain), np.array(z_chain), grad_energy_evol, acceptance_rate_evol