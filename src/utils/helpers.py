# src/utils/helpers.py

import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)
        
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def check_sampler(sampler):
    valid_samplers = ['ula', 'mala']
    if sampler not in valid_samplers:
        raise ValueError('Please select a valid Sampling method.')

def makedir(dir_name):
    if os.path.exists(os.path.join(dir_name, 'ckpt')):
        print('Output directory already exists')
    else:
        os.makedirs(os.path.join(dir_name, 'ckpt'))
        os.makedirs(os.path.join(dir_name, 'samples'))
        os.makedirs(os.path.join(dir_name, 'chains'))

def save_model(dir_name, epoch, model_name, E, optE, lr_scheduleE, G, optG, lr_scheduleG):
    save_dict = {
        'epoch': epoch,
        'GMM': E.state_dict(),
        'optGMM': optE.state_dict(),
        'lr_scheduleGMM': lr_scheduleE.state_dict(),
        'G': G.state_dict(),
        'optG': optG.state_dict(),
        'lr_scheduleG': lr_scheduleG.state_dict()
    }
    torch.save(save_dict, f'{dir_name}/ckpt/{model_name}.pth')

def load_model(dir_name, model_name, GMM, G):
        total_state_dict = torch.load(f'{dir_name}/ckpt/{model_name}.pth')
        GMM.load_state_dict(total_state_dict['GMM'])
        GMM.eval().to(device)
        G.load_state_dict(total_state_dict['G'])
        G.eval().to(device)
        return GMM, G

def sample_p_data(data, num_gen_samples):
    sample = data[torch.LongTensor(num_gen_samples).random_(0, data.size(0))].detach()
    return sample.view(sample.shape[0], sample.shape[1], sample.shape[2] * sample.shape[3])

def check_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    if num_nans > 0:
        raise ValueError(f'There are {num_nans} NaN values in the tensor.')

def savefig(sampler, likelihood, model, samples, num_steps_post, step_size_post, pixel, name):

    grid_num = 10
    
    generated_priors = samples.view(grid_num*grid_num, 1, pixel, pixel).detach().cpu()
    grid = torchvision.utils.make_grid(generated_priors, nrow=grid_num, padding=0)
    
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'../visuals/{model}/{name}_{sampler}_{likelihood}_{num_steps_post}_{step_size_post}.png')
    plt.close()

def emd(set1, set2):
    """
    Compute the Earth Mover's Distance between two sets of functions.

    Args:
    set1 (torch.Tensor): A tensor of shape (M, N) where M is the number of functions in set1 and N is the number of points.
    set2 (torch.Tensor): A tensor of shape (P, N) where P is the number of functions in set2 and N is the number of points.

    Returns:
    float: The Earth Mover's Distance between the two sets of functions.
    """
    
    # Ensure that both sets are in the same device
    if set1.device != set2.device:
        set2 = set2.to(set1.device)

    # Compute pairwise distances between the points in set1 and set2
    # We use the Euclidean distance here as an example
    cost_matrix = torch.cdist(set1, set2, p=2)

    # Normalize cost_matrix as required by the ot.emd2 function
    cost_matrix = cost_matrix.cpu().numpy()

    # Uniform weights for the distributions
    a = torch.ones(set1.size(0)) / set1.size(0)
    b = torch.ones(set2.size(0)) / set2.size(0)

    a = a.numpy()
    b = b.numpy()

    # Compute the Earth Mover's Distance using POT
    emd_distance = ot.emd2(a, b, cost_matrix, numItermax=1000000)

    return emd_distance