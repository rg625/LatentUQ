# src/main.py

import os
import numpy as np
import torch
from models.gmm import GMM
from models.darcy_solver import DarcySolver
from sampling.sampler import Sampling
from utils.helpers import makedir, sample_p_data, get_lr
from utils.logging_setup import setup_logging

# Set the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the data
data_dir = '../../data'
y_values_path = os.path.join(data_dir, 'poisson_y.npy')
a_path = os.path.join(data_dir, 'poisson_a.npy')

y_values = torch.tensor(np.load(y_values_path)).double().to(device)
a = torch.tensor(np.load(a_path)).double().to(device)
time = torch.linspace(0., 1., 100).to(device)
pushforward = DarcySolver(time)
y_res = (y_values - y_values.mean(dim=0)).squeeze()

data = y_values.view(-1, 1, len(time), 1)
data = data[torch.randperm(data.size(0))]

# Configuration
sampler = 'ula'
likelihood = 'gauss'
n_components = 2

# Training parameters
n_iter = 1000
num_steps_post = 50
step_size_post = 5e-2

# Initialize models
GMM_model = GMM(n_components, 5, width=1., std=1.).to(device)
G_model = G1D(5, nu=0.5).to(device)

# Initialize the sampling runner
runner = Sampling(G_model, GMM_model, sampler, likelihood, pushforward, time)
runner.dataset = 'poisson_gp'
runner.log_likelihood_sigma = 0.01
runner.plot = False
runner.burn_in = False
runner.true_coeffs = a
runner.lrGMM = 1e-3
runner.lrG = 1e-3

# Train the models
z_after = runner.train(data, n_iter, 100, num_steps_post, step_size_post)

# Generate samples (optional)
# output_prior, output_posterior = runner.generate_samples(data, num_gen_samples=100, n_iter=n_iter, num_steps_post=num_steps_post, step_size_post=step_size_post, ckpt='final')