# -*- coding: utf-8 -*-
"""
Custom dataset classes. Input function u, output function y. 
DeepONetDataset additionally requires a domain x.
"""

import torch
from torch.utils.data import Dataset, Sampler

class BasicDataset(Dataset):
    def __init__(self, u, y, device="cpu"):
        self.u = u.to(device)
        self.y = y.to(device)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.u[idx], self.y[idx]
    
class DeepONetDataset(Dataset):
    def __init__(self, u, x, y, device="cpu"):
        """
        u (pytorch tensor) : num_fun_samples*len_u_i*dim(u_i(x))
        x (pytorch tensor) : num_loc_samples*dim(x)
        y (pytorch tensor) : num_fun_samples*num_loc_samples*dim(y(x_i))
        (The domain sample size num_loc_samples is assumed to be the same for each function sample pair.)
        """
        self.u = u.to(device)
        self.x = x.to(device)
        self.y = y.to(device)
        self.num_loc_samples = len(x)
        
    def __len__(self):
        return self.y.nelement()
    
    def __getitem__(self, idx):
        return self.u[idx//self.num_loc_samples], \
                self.x[idx%self.num_loc_samples], \
                self.y[idx//self.num_loc_samples, idx%self.num_loc_samples]

class DeepONetDataloader:
    def __init__(self, dataset, fun_batch_size, loc_batch_size, shuffle=True):
        self.dataset = dataset
        self.num_fun_samples, self.num_loc_samples = dataset.y.shape[:2]
        self.fun_batch_size = fun_batch_size
        self.loc_batch_size = loc_batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        N = len(self.dataset)
        batch_size = self.fun_batch_size*self.loc_batch_size
        return N//batch_size + N%batch_size # ceil(N/batch_size)
    
    def _generate_indices(self):
        """
        Creates a list of indices for (u,x,y(x)) samples
        """
        if self.shuffle:
            fun_indices = torch.randperm(self.num_fun_samples)
            loc_indices = torch.stack([torch.randperm(self.num_loc_samples) for i in range(self.num_fun_samples)])
        else:
            fun_indices = torch.arange(self.num_fun_samples)
            loc_indices = torch.stack([torch.arange(self.num_loc_samples) for i in range(self.num_fun_samples)])
        return fun_indices, loc_indices
    
    def __iter__(self):
        fun_indices, loc_indices = self._generate_indices()
        shuffled_batches = []
        for i in range(max(self.num_fun_samples//self.fun_batch_size,1)):
            start_i = i*self.fun_batch_size
            end_i = start_i + self.fun_batch_size
            for j in range(max(self.num_loc_samples//self.loc_batch_size,1)):
                start_j = j*self.loc_batch_size
                end_j = start_j + self.loc_batch_size
                fun_idx_batch = fun_indices[start_i:end_i]
                loc_idx_batch = loc_indices[fun_idx_batch, start_j:end_j]
                
                shuffled_batches.append((self.dataset.u[fun_idx_batch], 
                                         self.dataset.x[fun_idx_batch.unsqueeze(1), loc_idx_batch],
                                         self.dataset.y[fun_idx_batch.unsqueeze(1), loc_idx_batch]))
        return iter(shuffled_batches)
    
if __name__=="__main__":
    # a small sanity check for correct implementation
    torch.manual_seed(0)
    num_fun_samples = 4
    u = torch.arange(num_fun_samples*5).view(num_fun_samples,-1)
    num_loc_samples = 10
    x = torch.linspace(0.,1.,num_loc_samples)
    y = torch.arange(num_fun_samples*num_loc_samples).view(num_fun_samples, num_loc_samples)
    ds = DeepONetDataset(u,x,y)
    fun_batch_size, loc_batch_size = (2,3)
    dl = DeepONetDataloader(ds, fun_batch_size, loc_batch_size, shuffle=False)
    for i, (u_, x_, y_) in enumerate(dl):
        print(u_)
        print(x_)
        print(y_)