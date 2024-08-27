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
                 n_conv_layers=0,
                 use_dropout=False,
                 final_activation_trunk=True
                 ):
        """
        input_size_branch (int or tuple of ints): size of branch input u; e.g. if u is 2d array needs tuple (N_t,N_x)
        input_size_trunk (int): size of trunk input x
        branch_architecture (list): defines the size of each fully connected layer in branch net
        trunk_architecture (list): defines the size of each layer in trunk net
        num_out_channels (int): the size of the DeepONet output (1 for scalar outputs)
        activation_branch (element-wise function): activation function between branch layers
        activation_trunk (element-wise function): activation function between trunk layers
        n_von_layers (int): the number of convolution layers in the branch net
        use_dropout (bool): whether or not to use dropout for training
        final_activation_trunk (bool): whether or not to apply an activation function to the trunk output
        """
        
        assert branch_architecture[-1] == trunk_architecture[-1], "size of latent dimension (final layer) must be the same for trunk and branch net"
        if type(input_size_branch) == tuple:
            assert len(input_size_branch) == input_size_trunk, "the branch input needs as many dimensions as the size of the trunk input"
            # get total number of elements in branch input (e.g. N=N_t*N_x)
            n_elements_branch_input = 1
            for dim_size in input_size_branch:
                n_elements_branch_input *= dim_size
        else:
            n_elements_branch_input = input_size_branch
        super().__init__()
        
        # check if convolutional branch architecture is to be used
        use_convolution = n_conv_layers > 0
        if use_convolution:
            # create conv layers that successively double the number of channels and halve the size per channel
            # consider having input_size_branch either int for u(x) or tuple for u(t,x)
            # then possibly do flatten for fully connected net (n_conv_layers=0)
            if input_size_trunk == 1:
                pool = lambda i: torch.nn.MaxPool1d(kernel_size=2)
                conv_layer = lambda i: torch.nn.Conv1d(2**i, 2**(i+1), kernel_size=3, padding=1)
            elif input_size_trunk == 2:
                pool = lambda i: torch.nn.MaxPool2d(kernel_size=2)
                conv_layer = lambda i: torch.nn.Conv2d(2**i, 2**(i+1), kernel_size=3, padding=1)
            else:
                assert False, "only 2d trunk inputs supported atm"
            def init_conv_net():
                conv_layers = []
                for i in range(n_conv_layers):
                    conv_layers.append(conv_layer(i))
                    conv_layers.append(activation_branch)
                    conv_layers.append(pool(i))
                conv_net = torch.nn.Sequential(*conv_layers)
                return conv_net
        # intialise a branch net for each output dimension
        
        
        branch_nets = []
        for channel in range(num_out_channels):
            branch_layers = []
            if use_convolution:
                branch_layers.append(init_conv_net())
                # flatten channel dimension and produce a vector
                # of size 2^n*n_elements_branch_input/(2d)^n = n_elements_branch_input/2^d (doubling channels and 2-pooling in d dimensions)
                branch_layers.append(torch.nn.Flatten()) 
            else:
                branch_layers.append(torch.nn.Flatten())
            branch_layers.append(torch.nn.Linear(n_elements_branch_input//input_size_trunk**n_conv_layers, branch_architecture[0]))
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
        
        # final bias
        self.bias = torch.nn.Parameter(torch.zeros(1,1,num_out_channels))
        
        # initialise weights
        self._initialise_weights(activation_branch, activation_trunk)
        
        if use_convolution:
            self.forward = self.forward_conv
        else:
            self.forward = self.forward_fc
        
    
            
    def forward_conv(self, u, x):
        # u is tensor representing n-valued function evaluated at n_u points, with shape (no. of function samples, n_u*n)
        # x is tensor representing point in R^m, with shape (no.of function samples, no. of input points, m)
        u = u.unsqueeze(1) # add channel dim (axis 1) of size 1
        u = torch.stack([branch(u) for branch in self.branch], dim=1) # produces a latent vector for each output dimension (keeping batch as first dimension)
        x = self.trunk(x)
        # B: function batch; y: output channel; i: latent dimension; b: domain batch
        out = torch.einsum('Byi, Bbi -> Bby', u, x) + self.bias
        return out
    
    def forward_fc(self, u, x):
        # u is tensor representing n-valued function evaluated at n_u points, with shape (no. of function samples, n_u*n)
        # x is tensor representing point in R^m, with shape (no.of function samples, no. of input points, m)
        u = torch.stack([branch(u) for branch in self.branch], dim=1) # produces a latent vector for each output dimension (keeping batch as first dimension)
        x = self.trunk(x)
        # B: function batch; y: output channel; i: latent dimension; b: domain batch
        out = torch.einsum('Byi, Bbi -> Bby', u, x) + self.bias
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
    
    def _initialise_weights(self, activation_branch, activation_trunk):
        # initialises weights in each layer using a normal dist. Uses Kaiming He (spread~sqrt(1/n)) for ReLU and Xavier Glorot (spread~sqrt(2/n)) for sigmoid etc
        
        # branch weights
        if type(activation_branch) == type(torch.nn.ReLU()):
            # use Kaiming for ReLU
            for m in self.branch.modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif type(activation_branch) in ( type(torch.nn.Sigmoid()), type(torch.nn.Tanh()) ) :
            # use Glorot for sigmoid/tanh
            for m in self.branch.modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                    torch.nn.init.xavier_normal_(m.weight)
        
        # trunk weights
        if type(activation_trunk) == type(torch.nn.ReLU()):
            # use Kaiming for ReLU
            for m in self.trunk.modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif type(activation_trunk) in ( type(torch.nn.Sigmoid()), type(torch.nn.Tanh()) ) :
            # use Glorot for sigmoid/tanh
            for m in self.trunk.modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                    torch.nn.init.xavier_normal_(m.weight)