# -*- coding: utf-8 -*-
"""
Routines for training a pytorch model
"""

import torch
from itertools import zip_longest

def validation_step(model, loss_fn, val_data):
    # val_data = (u,x,y) for DON or (u,y) for FNO
    targets = val_data[-1]
    inputs = val_data[:-1]
    preds = model(*inputs)
    val_loss = loss_fn(preds, targets, *inputs)
    return val_loss.item()

def train_FNO(model,
          dataloader,
          dataset_val,
          iterations,
          loss_fn,
          lr=1e-3,
          weight_penalty=0.):
    # model is a FNO pytorch model
    # dataloader (DataLoader) : iterable of training dataset
    # dataset_val (tuple) : of the form (u_val, y_val)
    # iterations (int) : number of training epochs
    # loss_fn : loss function, returns singleton pytorch tensor
    # lr (float) : learning rate
    # weight_penalty (float) : L2 weight penalty
    
    def training_loop(model, optimizer, dataloader):
        loss_epoch = 0.
        
        for batch in dataloader:
            optimizer.zero_grad()
            u = batch[0]
            y = batch[1]
            
            # Compute forward pass, calculate loss and gradients and update parameters
            preds = model(u)
            loss = loss_fn(preds, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_epoch += loss.item()
        # Compute the mean loss over minibatches
        loss_epoch = loss_epoch/len(dataloader)
        return loss_epoch
    
    best_val_loss = validation_step(model, lambda preds, targets, inputs: loss_fn(preds, targets), dataset_val)
    best_epoch = 0
    best_model_params = model.state_dict()
    
    # set-up for training
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_penalty)
    loss_history = []
    
    # training loop
    for epoch in range(iterations):
        loss_epoch = training_loop(model, optimizer, dataloader)
        loss_history.append(loss_epoch)
        
        # compute validation loss
        val_loss = validation_step(model, lambda preds, targets, inputs: loss_fn(preds, targets), dataset_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_params = model.state_dict()
        if epoch%20==0:
            print("{}/{}".format(epoch,iterations), "Loss:", loss_epoch)
    
    # save the best model parameters
    print("Lowest validation error at epoch", best_epoch, ":", best_val_loss)
    model.load_state_dict(best_model_params)
    
    return loss_history

def train_DON(model,
          dataset, 
          dataset_val,
          iterations,
          loss_fn,
          batch_size_fun=None,
          batch_size_loc=None,
          lr=1e-3,
          weight_penalty=0.):
    # model is a DeepONet pytorch model
    # dataset (DeepONetDataset) : training dataset
    # dataset_val (tuple) : of the form (u_val, x_val, y_val)
    # iterations (int) : number of training epochs
    # loss_fn : loss function. arguments u, x, preds, targets, idx; returns singleton pytorch tensor
    # batch_size_fun, batch_size_loc (int) : size of minibatches for u and x resp. Default is no minibatching
    # lr (float) : learning rate
    # weight_penalty (float) : L2 weight penalty
    
    best_val_loss = validation_step(model, loss_fn, dataset_val)
    best_epoch = 0
    best_model_params = model.state_dict()
    
    n_fun_samples = len(dataset)
    n_loc_samples = len(dataset.x[0])
    
    if batch_size_fun is None:
        batch_size_fun = n_fun_samples
    if batch_size_loc is None:
        batch_size_loc = n_loc_samples 
    
    n_fun_batches = n_fun_samples // batch_size_fun
    n_loc_batches = n_loc_samples // batch_size_loc
    
    def training_loop(model, optimizer):
        loss_epoch = 0.
        
        # Create a list of indices for (u,x,y(x)) samples
        fun_indices = torch.randperm(n_fun_samples)
        loc_indices = torch.stack([torch.arange(n_loc_samples) for i in range(n_fun_samples)])#torch.stack([torch.randperm(n_loc_samples) for i in range(n_fun_samples)])
        for i in range(n_fun_batches):
            i_start = i*batch_size_fun; i_end = (i+1)*batch_size_fun
            for j in range(n_loc_batches):
                
                optimizer.zero_grad() # zero gradients
                
                # Generate random data batches of function samples and location samples
                j_start = j*batch_size_loc; j_end = (j+1)*batch_size_loc
                fun_idx_batch = fun_indices[i_start:i_end]
                loc_idx_batch = loc_indices[fun_idx_batch, j_start:j_end]
                u_batch = dataset.u[fun_idx_batch]
                x_batch = dataset.x[fun_idx_batch.unsqueeze(1), loc_idx_batch]
                y_batch = dataset.y[fun_idx_batch.unsqueeze(1), loc_idx_batch]
                
                # Compute forward pass and loss; update weights
                preds = model(u_batch, x_batch)
                u_x = u_batch[torch.arange(batch_size_fun).unsqueeze(1), loc_idx_batch] # get the points that correspond to x_batch and y_batch
                loss = loss_fn(preds, y_batch, u_x, x_batch)
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_epoch += loss.item()
        
        # Compute the mean loss over minibatches
        loss_epoch = loss_epoch/(n_fun_batches*n_loc_batches)
        return loss_epoch
    
    # set-up for training
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_penalty)
    loss_history = []
    
    # training loop
    for epoch in range(iterations):
        loss_epoch = training_loop(model, optimizer)
        loss_history.append(loss_epoch)
        model.eval()
        # compute validation loss
        val_loss = validation_step(model, loss_fn, dataset_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_params = model.state_dict()
        model.train()
        if epoch%100==0:
            print("{}/{}".format(epoch,iterations), "Loss:", loss_epoch)
    
    # save the best model parameters
    print("Lowest validation error at epoch", best_epoch, ":", best_val_loss)
    model.load_state_dict(best_model_params)
    
    return torch.tensor(loss_history)