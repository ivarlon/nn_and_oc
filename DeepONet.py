# -*- coding: utf-8 -*-
"""
Implements a DeepONet in Pytorch.
"""
import torch

class DeepONet(torch.nn.Module):
    def __init__(self, branch_architecture, trunk_architecture, num_out_channels=1, activation=torch.nn.ReLU):
        """
        branch_architecture (list): defines the size of each layer in branch net. The first element must be num_input_channels*num_input_points
        trunk_architecture (list): defines the size of each layer in trunk net
        num_out_channels (int): the size of the DeepONet output (1 for scalar outputs)
        activation (element-wise function): activation function between layers
        """
        
        assert branch_architecture[-1] == trunk_architecture[-1], "size of latent dimension (final layer) must be the same for trunk and branch net"
        
        super().__init__()
        
        # create a branch net for each output dimension
        branch_layers = [torch.nn.Flatten()]
        for l in range(1,len(branch_architecture)-1):
            branch_layers.append(torch.nn.Linear(branch_architecture[l-1],branch_architecture[l]))
            branch_layers.append(activation)
        branch_layers.append(torch.nn.Linear(branch_architecture[-2], branch_architecture[-1]))
        self.branch = torch.nn.ModuleList([torch.nn.Sequential(*branch_layers) for i in range(num_out_channels)])
        
        trunk_layers = []
        for l in range(1,len(trunk_architecture)-1):
            trunk_layers.append(torch.nn.Linear(trunk_architecture[l-1],trunk_architecture[l]))
            trunk_layers.append(torch.nn.ReLU())
        trunk_layers.append(torch.nn.Linear(trunk_architecture[-2], trunk_architecture[-1]))
        self.trunk = torch.nn.Sequential(*trunk_layers)
        
        
    def forward(self, u, x):
        # u is tensor representing n-valued function evaluated at n_u points, with shape (no. of function samples, n_u*n)
        # x is tensor representing point in R^m, with shape (no. of input points, m)
        u = torch.stack([branch(u) for branch in self.branch], dim=1) # produces a latent vector for each output dimension (keeping batch as first dimension)
        
        x = self.trunk(x)
        
        out = torch.einsum('Byi..., bi... -> Bby...', u, x)
        return out