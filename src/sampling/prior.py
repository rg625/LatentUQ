import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_prior(**kwargs):
    """
    Compute the logarithm of the prior distribution.
    """
    z = kwargs['z']
    means = kwargs['means']
    lower_cholesky = kwargs['lower_cholesky']
    weights = kwargs['weights']
    eps = kwargs.get('eps', 1e-6)

    log_weights = torch.log(weights)
    log_det_covs = torch.log(torch.diagonal(lower_cholesky, dim1=-2, dim2=-1)).sum(dim=-1)
    diff = z[:, None, :] - means[None, :, :]
    exp_term_square = torch.linalg.solve_triangular(lower_cholesky, diff.unsqueeze(-1), upper=False)
    log_tot = log_weights[None, :] - log_det_covs[None, :] - 0.5 * torch.einsum('...kji,...kji->...k', exp_term_square, exp_term_square)
    return torch.logsumexp(log_tot, dim=-1).to(device)