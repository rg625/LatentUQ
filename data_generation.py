import sys
import os
import yaml
import numpy as np
import torch
from src.models import DarcySolver
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.utils.true_sampling import sample_function
from src.utils.helpers import makedir
from scipy.stats import norm

def main():
    # Add the project root directory to the PYTHONPATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    # Load the configuration file from the parent directory
    config_path = os.path.join(project_root, 'LatentUQ/configs', 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract configuration values
    data_dir = config['data_dir']
    dataset = config['dataset']
    makedir(dataset)
    num_points = 5000

    # Set the device for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    x = torch.linspace(0., 1., 100, device=device)
    log_sigma = torch.log(torch.tensor(2)) + 0.2*torch.randn((num_points, 1), device=device)
    ell = F.relu(1 + 0.4*torch.randn((num_points, 1), device=device))

    a = sample_function(x, 
                        batch_size= num_points, 
                        log_sigma=log_sigma, 
                        ell=ell, 
                        nu=torch.tensor(0.5)*torch.ones(num_points, 1, device=device))

    print(a.shape)
    solver = DarcySolver(x, amplitude = 0.1)
    u = solver(a)
    print(u.shape)

    # Plot the source term and the solution for the first realization
    plt.figure(figsize=(12, 6))

    # Plot source term f(x)
    plt.subplot(1, 2, 1)
    plt.hist(ell.cpu().numpy(), bins=50, density=True, alpha=0.5)
    x_axis = np.arange(0., 3., 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, 1, 0.4), label='gaussian fit')
    plt.axvline(x = 1., color = 'k', label = 'mean')
    plt.title('length scale')
    plt.legend()

    # Plot solution u(x)
    plt.subplot(1, 2, 2)
    plt.hist(log_sigma.cpu().numpy(), bins=50, density=True, alpha=0.5)
    x_axis = np.arange(0., 1.5, 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, np.log(2), 0.2), label='gaussian fit')
    plt.axvline(x = np.log(2), color = 'k', label = 'mean')
    plt.title('log_sigma')
    plt.legend()

    plt.tight_layout()
    plt.savefig(dataset+'/original_data/parameters_distribution.png')
    plt.close()
    plt.gcf().clear()

    # Plot the source term and the solution for the first realization
    plt.figure(figsize=(12, 6))

    # Plot source term f(x)
    plt.subplot(1, 2, 1)
    for i in range(5):
        plt.plot(x.cpu().numpy(), a[i].cpu().numpy(), label=f'Batch {i+1}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Source Term f(x)')
    plt.legend()

    # Plot solution u(x)
    plt.subplot(1, 2, 2)
    for i in range(5):
        plt.plot(x.cpu().numpy(), u[i].cpu().numpy(), label=f'Batch {i+1}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution u(x)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(dataset+'/original_data/functional_distribution.png')
    plt.close()
    plt.gcf().clear()
    
    u_batch = u + config['sigma_y']*torch.randn_like(u, device=device)

    # Load the data
    np.save(f'{data_dir}/{dataset}_a.npy', a.cpu().numpy())
    np.save(f'{data_dir}/{dataset}_y.npy', u_batch.cpu().numpy())


if __name__ == "__main__":
    main()