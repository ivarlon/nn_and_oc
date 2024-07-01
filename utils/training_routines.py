# -*- coding: utf-8 -*-
"""
Routines for training a pytorch model
"""

import torch

def training_loop(model, optimizer, dataloader, boundary_condition, 
                  weight_boundary=1., 
                  weight_prior=1.,
                  train_adjoint=False):
    loss_fn = torch.nn.MSELoss()
    loss_t = 0.
    batch_size = 8
    num_batches = len(dataloader.dataset)//dataloader.batch_size
    
    for inputs_batch, targets_batch in dataloader:
        optimizer.zero_grad()
        
        
        preds = model(inputs_batch)
        
        #exp_x = torch.exp(torch.linspace(0.,1., len(inputs[0])))
        #exp_x_ux = torch.einsum('x, bxu->bxu', exp_x, inputs_batch)
        
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
    loss_t = loss_t/num_batches
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