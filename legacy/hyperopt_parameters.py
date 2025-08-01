import torch
import torch.nn as nn
import numpy as np
import random

def set_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_activation(activation_name):
    """Return the activation function based on the name"""
    activation_functions = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'Tanh': nn.Tanh(),
        'GELU': nn.GELU(),
        'ELU': nn.ELU(),
        'Sigmoid': nn.Sigmoid(),
        'SiLU': nn.SiLU()  # Also known as Swish
    }
    
    if activation_name in activation_functions:
        return activation_functions[activation_name]
    else:
        print(f"Warning: Activation '{activation_name}' not found, using ReLU")
        return nn.ReLU()
