# -*- coding: utf-8 -*-
"""
Routines for training a pytorch model
"""

import torch
from itertools import zip_longest

def training_loop(model, optimizer, dataloader, loss_fn, weights_losses):
    loss_epoch = 0.
    
    for batch in dataloader:
        optimizer.zero_grad()
        inputs_batch = batch[:-1]
        targets_batch = batch[-1]
        preds = model(*inputs_batch)
        
        loss = loss_fn(preds.flatten(), targets_batch.flatten())
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_epoch += loss.item()
    # Compute the mean loss over minibatches
    loss_epoch = loss_epoch/len(dataloader.dataset)
    return loss_epoch

def training_loop_physics(model, optimizer, dataloader_list, loss_fn_list, weights_losses):
    loss_epoch = 0.
    loss_fn_data, loss_fn_physics = loss_fn_list
    weight_data, weight_physics = weights_losses
    for batch_data, batch_physics in zip_longest(*dataloader_list, fillvalue=None):
        optimizer.zero_grad()
        if batch_physics is None:
            # If all physics data has been looped over, finish looping over rest of data
            inputs_data = batch_data[:-1]
            targets_data = batch_data[-1]
            preds_data = model(*inputs_data)
            loss_batch = weight_data*loss_fn_data(preds_data.flatten(), targets_data.flatten())
        elif batch_data is None:
            preds_physics = model(*batch_physics)
            loss_batch = weight_physics*loss_fn_physics(*batch_physics, preds_physics)
        else:
            # data loss
            inputs_data = batch_data[:-1]
            targets_data = batch_data[-1]
            preds_data = model(*inputs_data)
            loss_data = loss_fn_data(preds_data.flatten(), targets_data.flatten())
            # physics loss
            preds_physics = model(*batch_physics)
            loss_physics = loss_fn_physics(*batch_physics, preds_physics)
            loss_batch = weight_data*loss_data + weight_physics*loss_physics
        loss_batch.backward(retain_graph=True)
        optimizer.step()
        loss_epoch += loss_batch.item()
    # Compute the mean loss over minibatches
    loss_epoch = loss_epoch/max(len(dataloader_list[0].dataset), len(dataloader_list[1].dataset))
    return loss_epoch

def train(model, 
          dataloader, 
          iterations,
          loss_fn,
          weights_losses=1.,
          lr=1e-3,
          weight_penalty=0., 
          boundary_condition=1.):
    
    # Determine whether a physics-informed model is to be trained or not (then dataloader=[dl_data,dl_physics_data])
    train_physics = type(dataloader) == list
    if not train_physics:
        loop = training_loop
    else:
        loop = training_loop_physics
    
    # set-up for training
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_penalty)
    loss_history = []
    
    # training loop
    for epoch in range(iterations):
        loss_epoch = loop(model, optimizer, dataloader, loss_fn, 
                          weights_losses=weights_losses)
        loss_history.append(loss_epoch)
        if epoch%10==0:
            print("{}/{}".format(epoch,iterations), "Loss:", round(loss_epoch,7))
    
    return torch.tensor(loss_history)