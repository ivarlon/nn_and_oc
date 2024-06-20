# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:50:52 2024

@author: L390
"""

import torch

class DeepONet(torch.nn.Module):
    def __init__(self, trunk_architecture, branch_architecture):
        super().__init__()
        
        self.trunk =  torch.randn(size=trunk_architecture) 
        trunk_normalisation = torch.zeros(len(self.trunk))
        for layer in range(len(self.trunk)):
            trunk_normalisation[layer] = (2./len(self.trunk[layer]))**0.5
        
        #self.trunk *= trunk_normalisation
        
        self.branch = torch.randn(size=branch_architecture)
        branch_normalisation = torch.zeros(len(self.branch))
        for layer in range(len(self.branch)):
            trunk_normalisation[layer] = (2./len(self.branch[layer]))**0.5
        
        #self.branch *= branch_normalisation
        
        self.trunk = torch.nn.Parameter( self.trunk )
        self.branch = torch.nn.Parameter( self.branch )
        
        
    def forward(self, u, x):
        for layer in self.trunk:
            print(layer.shape)
            u = torch.einsum('ij, i...-> j...', layer, u)
            u = torch.relu(u)
        
        for layer in self.branch:
            x = torch.einsum('ij, ki...-> kj...', layer, x)
            x = torch.relu(x)
        
        return torch.einsum('i..., ki..., -> k...', u, x)
