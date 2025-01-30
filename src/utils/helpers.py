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
        'netE': E.state_dict(),
        'optE': optE.state_dict(),
        'lr_scheduleE': lr_scheduleE.state_dict(),
        'netG': G.state_dict(),
        'optG': optG.state_dict(),
        'lr_scheduleG': lr_scheduleG.state_dict()
    }
    torch.save(save_dict, f'{dir_name}/ckpt/{model_name}.pth')

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