# -*- coding: utf-8 -*-
"""
Implements a DeepONet in Pytorch.
"""
import torch

class DeepONet(torch.nn.Module):
    def __init__(self,
                 input_size_branch,
                 input_size_trunk,
                 branch_architecture,
                 trunk_architecture,
                 num_out_channels=1,
                 activation_branch=torch.nn.ReLU(),
                 activation_trunk=torch.nn.Sigmoid(),
                 use_dropout=False,
                 final_activation_trunk=True
                 ):
        """
        input_size_branch (int): size of *flattened* branch input u; equal to num_input_channels*num_input_points
        input_size_trunk (int): size of trunk input x
        branch_architecture (list): defines the size of each layer in branch net
        trunk_architecture (list): defines the size of each layer in trunk net
        num_out_channels (int): the size of the DeepONet output (1 for scalar outputs)
        activation_branch (element-wise function): activation function between branch layers
        activation_trunk (element-wise function): activation function between trunk layers
        final_activation_trunk (bool): whether or not to apply an activation function to the trunk output
        """
        
        assert branch_architecture[-1] == trunk_architecture[-1], "size of latent dimension (final layer) must be the same for trunk and branch net"
        
        super().__init__()
        
        # intialise a branch net for each output dimension
        branch_nets = []
        for channel in range(num_out_channels):
            branch_layers = [torch.nn.Flatten()]
            branch_layers.append(torch.nn.Linear(input_size_branch, branch_architecture[0]))
            if len(branch_architecture)>1:
                for l in range(1, len(branch_architecture)):
                    branch_layers.append(activation_branch)
                    if use_dropout:
                        branch_layers.append(torch.nn.Dropout(p=0.5))
                    branch_layers.append(torch.nn.Linear(branch_architecture[l-1],branch_architecture[l]))
            branch_nets.append( torch.nn.Sequential(*branch_layers) )
        self.branch = torch.nn.ModuleList(branch_nets)
        
        # initialise trunk net
        trunk_layers = []
        trunk_layers.append(torch.nn.Linear(input_size_trunk, trunk_architecture[0]))
        if len(trunk_architecture)>1:
            for l in range(1,len(trunk_architecture)):
                trunk_layers.append(activation_trunk)
                if use_dropout:
                    trunk_layers.append(torch.nn.Dropout(p=0.5))
                trunk_layers.append(torch.nn.Linear(trunk_architecture[l-1],trunk_architecture[l]))
        if final_activation_trunk:
            trunk_layers.append(activation_trunk)
        self.trunk = torch.nn.Sequential(*trunk_layers)
        
        self.bias = torch.nn.Parameter(torch.zeros(1,1,num_out_channels))
        
        # set forward method so that trunk input tensor x has same first dimension as branch input u
        self.trunk_input_shares_first_dimension_with_branch_input(True)
        
    def trunk_input_shares_first_dimension_with_branch_input(self, share):
        """
        Alternates the forward method depending on whether
        trunk input tensor x has a first dimension corresponding to batch dimension
        of branch input (useful if domain x differs for different branch input u)
        """
        if share == True:
            self.forward = self.forward_share
        else:
            self.forward = self.forward_dont_share
            
    def forward_share(self, u, x):
        # u is tensor representing n-valued function evaluated at n_u points, with shape (no. of function samples, n_u*n)
        # x is tensor representing point in R^m, with shape (no.of function samples, no. of input points, m)
        u = torch.stack([branch(u) for branch in self.branch], dim=1) # produces a latent vector for each output dimension (keeping batch as first dimension)
        x = self.trunk(x)
        # B: function batch; y: output channel; i: latent dimension; b: domain batch
        out = torch.einsum('Byi..., Bbi... -> Bby...', u, x) + self.bias
        return out
    
    def forward_dont_share(self, u, x):
        # forward method for trunk input x the same for every branch sample
        # u is tensor representing n-valued function evaluated at n_u points, with shape (no. of function samples, n_u*n)
        # x is tensor representing point in R^m, with shape (no. of input points, m)
        u = torch.stack([branch(u) for branch in self.branch], dim=1) # produces a latent vector for each output dimension (keeping batch as first dimension)
        x = self.trunk(x)
        # B: function batch; y: output channel; i: latent dimension; b: domain batch
        out = torch.einsum('Byi..., bi... -> Bby...', u, x) + self.bias
        return out