import sys
import os
import yaml
import numpy as np
import torch
from src.models import GMM, DarcySolver, OrthogonalBasis, CosineBasis, Integrator
from src.sampling import Sampling
from src.utils import makedir, sample_p_data, get_lr
from src.utils import setup_logging

def main(config_path):
    # Add the project root directory to the PYTHONPATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    # Load the configuration file from the provided path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract configuration values
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    channels = config['channels']
    gmm_dim = config['gmm_dim']
    n_components = config['gmm_components']
    sampler = config['sampler']
    likelihood = config['likelihood']
    num_steps = config['num_steps']
    step_size = config['step_size']
    epochs = config['epochs']
    model = config['model']
    dataset = config['dataset']
    num_expansions = config['num_expansions']
    # Set the device for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the data
    y_values_path = os.path.join(data_dir, f'{dataset}_y.npy')
    a_path = os.path.join(data_dir, f'{dataset}_a.npy')

    y_values = torch.tensor(np.load(y_values_path)).double().to(device)
    a = torch.tensor(np.load(a_path)).double().to(device)
    time = torch.linspace(0., 1., 100).to(device)
    if dataset == 'poisson':
        pushforward = DarcySolver(time)
    elif dataset == 'ode':
        pushforward = Integrator(time)
        
    data = y_values.view(-1, 1, len(time), channels)
    data = data[torch.randperm(data.size(0))]

    # Initialize models
    GMM_model = GMM(n_components, gmm_dim, width=1., std=1.).to(device)
    if model == 'DeepONet':
        G_model = OrthogonalBasis(gmm_dim, num_expansions, nu=0.5).to(device)
        print(f'Using {config["model"]} model')
    elif model == 'cosine':
        G_model = CosineBasis(gmm_dim, nu=0.5).to(device)
        print(f'Using {config["model"]} model')
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    # Initialize the sampling runner
    runner = Sampling(G_model, GMM_model, sampler, likelihood, pushforward, time)
    runner.dataset = dataset
    runner.log_likelihood_sigma = config['sigma_y']
    runner.plot = config['plot']
    runner.true_coeffs = a
    runner.lrGMM = config['lr_GMM']
    runner.lrG = config['lr_G']

    # Train the models
    runner.train(data = data, n_iter = epochs, batch_size = batch_size, num_steps_post = num_steps, step_size_post = step_size)

    # Generate samples (optional)
    # output_prior, output_posterior = runner.generate_samples(data, num_gen_samples=100, n_iter=n_iter, num_steps_post=num_steps_post, step_size_post=step_size_post, ckpt='final')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)