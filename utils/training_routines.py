# -*- coding: utf-8 -*-
"""
Routines for training a pytorch model
"""

import torch
torch.set_default_dtype(torch.float32) # all tensors are float32
from itertools import zip_longest

def validation_step(model, loss_fns, val_data):
    # val_data = (u,x,y) for DON or (u,y) for FNO
    targets = val_data[-1]
    inputs = val_data[:-1]
    preds = model(*inputs)
    val_loss = 0.
    for loss_fn in loss_fns:
        val_loss += loss_fn(preds, targets, *inputs)
    return val_loss.item()

def train_FNO(model,
          dataloader,
          dataset_val,
          iterations,
          loss_fn,
          device='cpu',
          lr=1e-3,
          weight_penalty=0.,
          print_every=None):
    # model is a FNO pytorch model
    # dataloader (DataLoader) : iterable of training dataset
    # dataset_val (tuple) : of the form (u_val, y_val)
    # iterations (int) : number of training epochs
    # loss_fn : loss function, returns singleton pytorch tensor
    # lr (float) : learning rate
    # weight_penalty (float) : L2 weight penalty
    # print_every (None or int) : prints the training loss for every print_every epoch, or no print if print_every==None
    
    def training_loop(model, optimizer, dataloader):
        loss_epoch = 0.
        
        for batch in dataloader:
            optimizer.zero_grad()
            u = batch[0].to(device)
            y = batch[1].to(device)
            
            # Compute forward pass, calculate loss and gradients and update parameters
            preds = model(u)
            loss = loss_fn(preds, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_epoch += loss.item()
        # Compute the mean loss over minibatches
        loss_epoch = loss_epoch/len(dataloader)
        return loss_epoch
    
    best_val_loss = validation_step(model, (lambda preds, targets, inputs: loss_fn(preds, targets),), dataset_val)
    best_epoch = 0
    best_model_params = model.state_dict()
    
    # set-up for training
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_penalty)
    loss_history_train = []
    loss_history_val = []
    
    # training loop
    for epoch in range(iterations):
        loss_epoch = training_loop(model, optimizer, dataloader)
        loss_history_train.append(loss_epoch)
        
        # compute validation loss
        val_loss = validation_step(model, (lambda preds, targets, inputs: loss_fn(preds, targets),), dataset_val)
        loss_history_val.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_params = model.state_dict()
        if print_every:
            if epoch%print_every==0:
                print("{}/{}".format(epoch,iterations), "Loss:", loss_epoch)
    
    # save the best model parameters
    print("Lowest validation error at epoch", best_epoch, ":", best_val_loss)
    model.load_state_dict(best_model_params)
    
    return torch.tensor(loss_history_train), torch.tensor(loss_history_val)

def train_DON(model,
          dataset, 
          dataset_val,
          iterations,
          loss_fn_data,
          loss_fn_physics,
          optimizer="adam",
          device='cpu',
          batch_size_fun=None,
          batch_size_loc=None,
          lr=1e-3,
          weight_penalty=0.,
          print_every=None):
    # model is a DeepONet pytorch model
    # dataset (DeepONetDataset) : training dataset
    # dataset_val (tuple) : of the form (u_val, x_val, y_val)
    # iterations (int) : number of training epochs
    # loss_fn : loss function. arguments u, x, preds, targets, idx; returns singleton pytorch tensor
    # batch_size_fun, batch_size_loc (int) : size of minibatches for u and x resp. Default is no minibatching
    # lr (float) : learning rate
    # weight_penalty (float) : L2 weight penalty
    # print_every (None or int) : prints the training loss for every print_every epoch, or no print if print_every==None
    
    best_val_loss = validation_step(model, (loss_fn_data, loss_fn_physics), dataset_val)
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
        loss_data_epoch = 0.
        loss_physics_epoch = 0.
        
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
                u_batch = dataset.u[fun_idx_batch].to(device)
                x_batch = dataset.x[fun_idx_batch.unsqueeze(1), loc_idx_batch].to(device)
                
                y_batch = dataset.y.flatten(start_dim=1)[fun_idx_batch.unsqueeze(1), loc_idx_batch].view(batch_size_fun, *dataset.y.shape[1:]).to(device)
                
                # Compute forward pass and loss; update weights
                preds = model(u_batch, x_batch)
                
                u_x = u_batch.flatten(start_dim=1)[torch.arange(batch_size_fun).unsqueeze(1), loc_idx_batch].view(batch_size_fun, *dataset.u.shape[1:]) # get the points that correspond to x_batch and y_batch
                
                loss_data = loss_fn_data(preds, y_batch, u_x, x_batch)
                loss_physics = loss_fn_physics(preds, y_batch, u_x, x_batch)
                loss = loss_data + loss_physics
                loss.backward(retain_graph=True)
                
                optimizer.step()
                
                loss_epoch += loss.item()
                loss_data_epoch += loss_data.item()
                loss_physics_epoch += loss_physics.item()
        
        # Compute the mean loss over minibatches
        loss_epoch = loss_epoch/(n_fun_batches*n_loc_batches)
        loss_data_epoch = loss_data_epoch/(n_fun_batches*n_loc_batches)
        loss_physics_epoch = loss_physics_epoch/(n_fun_batches*n_loc_batches)
        return loss_epoch, loss_data_epoch, loss_physics_epoch
    
    # set-up for training
    model.train()
    if optimizer=="RMSprop":
        optim = torch.optim.RMSprop(params=model.parameters(), lr=lr, weight_decay=weight_penalty, momentum=0.9)
    else:
        optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_penalty)
    loss_history_train = []
    loss_data_history = []
    loss_physics_history = []
    loss_history_val = []
    
    # training loop
    for epoch in range(iterations):
        loss_epoch, loss_data_epoch, loss_physics_epoch = training_loop(model, optim)
        loss_history_train.append(loss_epoch)
        loss_data_history.append(loss_data_epoch)
        loss_physics_history.append(loss_physics_epoch)
        model.eval()
        # compute validation loss
        val_loss = validation_step(model, (loss_fn_data, loss_fn_physics), dataset_val)
        loss_history_val.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_params = model.state_dict()
        model.train()
        if print_every:
            if epoch%print_every==0:
                print("{}/{}".format(epoch,iterations), "Loss:", loss_epoch)
    
    # save the best model parameters
    print("Lowest validation error at epoch", best_epoch, ":", best_val_loss)
    model.load_state_dict(best_model_params)
    
    return torch.tensor(loss_history_train), torch.tensor(loss_data_history), torch.tensor(loss_physics_history), torch.tensor(loss_history_val)