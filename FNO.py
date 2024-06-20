# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:57:29 2024

Fourier Neural Operator
solving y' = y + u

Input to FNO is an evenly discretised source function u_i = u(x_i) 
in N points, where N is a power of 2, which lets us use the fast Fourier transform
"""

import torch
#from tqdm import trange

"""
Need:
    evenly discretised domain (here x_i = i/(N-1), i = 0, 1, ..., N-1)
    u_i = u(x_i)
"""

class FNO(torch.nn.Module):
    """
    Fourier neural operator.
    Input arguments:
        n_layers (int) : number of layers of NN
        N (int) : number of grid points. N should be a power of 2
        d_u (int) : dimensionality of input u(x_i)
        d_v (int) : dimensionality of lifted input
        k_max (int) : max. wave number used by FNO
    """
    
    def __init__(self, n_layers, N, d_u, d_v, k_max=6):
        super().__init__()
        
        self.n_layers = n_layers
        self.N = N
        self.k_max = min(k_max, N) # FNO ignores wave numbers above k_max. k_max should not exceed no. of points N
        
        # initial lifting operator
        self.lift = torch.nn.Linear(d_u, d_v)
        
        # final projection operator
        self.proj = torch.nn.Linear(d_v, d_u)
        
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
        # u is pytorch tensor of size (n_batches, N, d_u)
        v = self.lift(u)
        for l in range(self.n_layers):
            v_ = torch.fft.rfft(v,dim=1)[:,:self.k_max] # do real FT over rows, truncate at k_max
            v_ = torch.einsum('kKj,bkj...->bkK...', self.kernel_params[l], v_) # multiply w/ F. kernel
            v_ = torch.fft.irfft(v_, dim=1, n=self.N) # inv. FFT returns tensor of length N
            v = torch.einsum('ij,bkj->bki', self.W[l], v) + self.bias[l] + v_
            v = self.activation(v)
        v = self.proj(v)
        return v
    

def training_loop(model, optimizer, inputs, targets, boundary_condition, 
                  weight_boundary=1., 
                  weight_prior=1.,
                  train_adjoint=False):
    loss_fn = torch.nn.MSELoss()
    loss_t = 0.
    batch_size = 8
    n_batches = len(inputs)//batch_size
    
    for batch in range(n_batches):
        optimizer.zero_grad()
        
        targets_batch = targets[batch_size*batch:batch_size*(batch+1)]
        inputs_batch = inputs[batch_size*batch:batch_size*(batch+1)]
        
        preds = model(inputs_batch)
        
        exp_x = torch.exp(torch.linspace(0.,1., len(inputs[0])))
        exp_x_ux = torch.einsum('x, bxu->bxu', exp_x, inputs_batch)
        
        loss_interior = loss_fn(preds.ravel(), targets_batch.ravel())
        if train_adjoint:
            # constrain terminal prediction
            loss_boundary = torch.mean( torch.mean( (preds[:,-1,:] - boundary_condition)**2 ) )
        else:
            # constrain initial condition
            loss_boundary = torch.mean( torch.mean( (preds[:,0,:] - boundary_condition)**2 ) )
        #loss_prior = torch.mean( ( preds - 1/exp_x*model(exp_x_ux) )**2 )
        loss = loss_interior + weight_boundary*loss_boundary #+ weight_prior*loss_prior
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_t += loss.item()
    loss_t = loss_t/n_batches
    return loss_t

def train(model, 
          inputs, 
          targets, 
          iterations,
          lr=1e-3,
          weight_penalty=0.,
          weight_boundary=1., 
          weight_prior=1., 
          boundary_condition=1.,
          train_adjoint=False):
    
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_penalty)
    loss_history = []
    
    for epoch in range(iterations):
        loss_t = training_loop(model, optimizer, inputs, targets, 
                               boundary_condition=boundary_condition,
                               train_adjoint=train_adjoint,
                               weight_boundary=weight_boundary, weight_prior=weight_prior)
        loss_history.append(loss_t)
    return torch.tensor(loss_history)