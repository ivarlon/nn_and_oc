# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:57:29 2024

Fourier Neural Operator
solving y' = y + u

Input to FNO is an evenly discretised source function u_i = u(x_i) 
in N points, where N is a power of 2, which lets us use the fast Fourier transform
"""

import torch


class FNO(torch.nn.Module):
    """
    Fourier neural operator.
    Input arguments:
        n_layers (int) : number of layers of NN
        N (int) : number of grid points. N should be a power of 2
        d_u (int) : dimensionality of input u(x_i)
        d_v (int) : dimensionality of lifted input
        d_y (int) : dimensionality of output
        k_max (int) : max. wave number used by FNO
    """
    
    def __init__(self, n_layers, N, d_u, d_v, d_y=None, k_max=6):
        super().__init__()
        
        self.n_layers = n_layers
        self.N = N
        self.k_max = min(k_max, N) # FNO ignores wave numbers above k_max. k_max should not exceed no. of points N
        if d_y is None:
            d_y = d_u
        
        
        # initial lifting operator
        self.lift = torch.nn.Linear(d_u, d_v)
        
        # final projection operator
        self.proj = torch.nn.Linear(d_v, d_y)
        
        # create a Fourier kernel for each k_i, i = 0, ..., N-1
        self.kernel_params = torch.nn.Parameter( torch.normal(mean=0.,
                            std=(2/self.k_max)**0.5,
                            size=(n_layers, self.k_max, d_v, d_v),
                            dtype=torch.complex64,
                            requires_grad=True
                            ) )
        
        # local linear operators --> W(x)*u(x)
        self.W = torch.nn.Parameter( torch.normal(mean=0.,
                              std=1.0,
                              size=(n_layers, d_v, d_v)) )
        
        # bias functions --> u(x) + b(x)
        self.bias = torch.nn.Parameter( torch.zeros(size=(n_layers, self.N, d_v)) )
        
        # activation function
        self.activation = torch.nn.ReLU()
        
        
        
    def forward(self, u):
        # calculates forward pass u --> NN(u)
        # u is pytorch tensor of size (n_batches, (size of input dimensions), d_u)
        u = u.flatten(start_dim=1, end_dim=-2) # flattens input dimensions
        v = self.lift(u)
        for l in range(self.n_layers):
            v_ = torch.fft.rfft(v,dim=1)[:,:self.k_max] # do real FT over rows, truncate at k_max
            v_ = torch.einsum('kKj,bkj...->bkK...', self.kernel_params[l], v_) # multiply w/ F. kernel
            v_ = torch.fft.irfft(v_, dim=1, n=self.N) # inv. FFT returns tensor of length N
            v = torch.einsum('ij,bkj->bki', self.W[l], v) + self.bias[l] + v_
            v = self.activation(v)
        v = self.proj(v)
        return v
