# -*- coding: utf-8 -*-
"""
Custom dataset classes. Input function u, output function y. 
DeepONetDataset additionally requires a domain x.
"""

import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, u, y):
        self.u = u
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.u[idx], self.y[idx]
    
class DeepONetDataset(Dataset):
    def __init__(self, u, x, y):
        """
        u (pytorch tensor) : num_fun_samples*len_u_i*dim(u_i(x))
        x (pytorch tensor) : num_loc_samples*dim(x)
        y (pytorch tensor) : num_fun_samples*num_loc_samples*dim(y(x_i))
        """
        self.u = u
        self.x = x
        self.y = y
        self.num_loc_samples = len(x)
        
    def __len__(self):
        return self.y.nelement()
    
    def __getitem__(self, idx):
        return self.u[idx//self.num_loc_samples], \
                self.x[idx%self.num_loc_samples], \
                self.y[idx//self.num_loc_samples, idx%self.num_loc_samples]