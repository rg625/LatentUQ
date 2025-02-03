import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian_likelihood(**kwargs):
    """
    Compute Gaussian likelihood between true and predicted data.
    """
    x = kwargs['x'].to(device)
    x_hat = kwargs['x_hat'].reshape(x.shape).to(device)
    sigma = torch.tensor(kwargs.get('sigma', 1.)).to(device)
    
    sigma_sq = sigma ** 2
    diff = (x - x_hat).unsqueeze(-1)
    return -torch.einsum('...ij,...ij->...j', diff, diff).squeeze() / (2. * sigma_sq)