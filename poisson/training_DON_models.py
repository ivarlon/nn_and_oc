# -*- coding: utf-8 -*-
"""
Trains Deep Operator Networks (DeepONets) to solve Poisson equation

The control/source is transformed before being input into the network:
    u -> u + u_shift -> log(u + u_shift)
This reduces the magnitude of u including the variance, and allows for better training.
"""

# for saving data
import sys
import os
import pickle
import time # to measure training time

# import numerical libraries
import numpy as np
import torch

# import custom libraries and functions
from DeepONet import DeepONet
from utils.training_routines import train_DON
from CustomDataset import *
from generate_data_poisson import generate_data, augment_data

cuda = 2 # 1,2,3

# seed pytorch RNG
seed = 12
torch.manual_seed(seed)

if torch.cuda.is_available():
    print("Using CUDA", cuda)
    device = torch.device("cuda:{}".format(cuda))
else:
    print("Using CPU\n")
    device = torch.device("cpu")

# create data directory to store models and results
data_dir_name = 'state_experiments_DON'
problem_dir_name = "poisson"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, problem_dir_name, data_dir_name)
if not os.path.exists(data_dir):
        os.makedirs(data_dir)


N = 64 # number of points along each direction

# size of domain
L = 1.

# boundary conditions
BCs = [torch.zeros(N) for i in range(4)] # Dirichlet boundary conditions on state

n_models = 3

################################
# Generate train and test data #
################################

n_train = 5000 # no. of training samples
n_test = 500 # no. of test samples
n_val = 400 # no. of training samples
batch_size_fun = 50 # minibatch size during SGD
batch_size_loc = N**2 # no. of minibatch domain points. Get worse performance when not using entire domain :/

u_max = 80. # maximum amplitude of control
u_shift = 100. # we input the log of control to NN, : u + u_shift > -u_max + u_shift > 0
generate_data_func = lambda n_samples: generate_data(N,
                  L,
                  BCs=[bc.numpy() for bc in BCs],
                  n_samples=n_samples,
                  u_max=u_max)

# generate different data for different models?
different_data = True
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data_func(n_train-2000)
        augment_data(data, n_augmented_samples=2000, n_combinations=5, max_coeff=2)
        data["r"].requires_grad = True
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data_func(n_train-2000)
    augment_data(data, n_augmented_samples=2000, n_combinations=5, max_coeff=2)
    data["r"].requires_grad = True
    train_data = n_models*[data]

# generate test and validation data
test_data = generate_data_func(n_test-200)
augment_data(test_data, n_augmented_samples=200, n_combinations=5, max_coeff=2)
test_data["r"].requires_grad = True
u_test = (test_data["u"] + u_shift).log(); r_test = test_data["r"]; y_test = test_data["y"]

val_data = generate_data_func(n_val-200)
augment_data(val_data, n_augmented_samples=200, n_combinations=5, max_coeff=2)
val_data["r"].requires_grad = True
dataset_val = ((val_data["u"] + u_shift).log().to(device), val_data["r"].to(device), val_data["y"].to(device))

################
# physics loss #
################

def PDE_interior(u,x,y):
    # u is size (n_fun_samples, N^2)
    # x is size (n_fun_samples, N^2, 2)
    # y is size (n_fun_samples, N^2)
    dy = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    dy_xx = torch.autograd.grad(outputs=dy[...,0], inputs=x, grad_outputs=torch.ones_like(dy[...,0]), create_graph=True, retain_graph=True)[0]
    dy_xx = dy_xx[...,0]
    dy_yy = torch.autograd.grad(outputs=dy[...,1], inputs=x, grad_outputs=torch.ones_like(dy[...,1]), create_graph=True, retain_graph=True)[0]
    dy_yy = dy_yy[...,1]
    
    dy_xx = dy_xx.view(y.shape[0], N, N)
    dy_yy = dy_yy.view(y.shape[0], N, N)
    
    laplacian_transformed = torch.log(-(dy_xx + dy_yy) + u_shift) # transform laplacian in same way as control u
    
    return laplacian_transformed[:,1:-1,1:-1] - u.view_as(dy_xx)[:,1:-1,1:-1]

def physics_loss(u, x, y, BCs, weight_BC=1.):
    # y = y(x;u) is output of DeepONet, tensor of shape (n_samples, n_domain_points, dim(Y))
    # x is input tensor and has shape (n_samples, n_domain_points, dim(X))
    interior_loss = (PDE_interior(u, x, y)**2).mean()
    #print(interior_loss)
    n_fun_samples = y.shape[0]
    y_reshaped = y.view(n_fun_samples, N, N) # (n_samples, N_t, N_x)
    
    BC_loss = torch.nn.MSELoss()(y_reshaped[:,:,0], BCs[0].repeat(n_fun_samples,1)) \
        + torch.nn.MSELoss()(y_reshaped[:,-1,:], BCs[1].repeat(n_fun_samples,1)) \
        + torch.nn.MSELoss()(y_reshaped[:,:,-1], BCs[2].repeat(n_fun_samples,1)) \
        + torch.nn.MSELoss()(y_reshaped[:,0,:], BCs[3].repeat(n_fun_samples,1))
    
    return interior_loss + weight_BC*BC_loss

#######################################
# Set up and train the various models #
#######################################

input_size_branch = (N, N)
input_size_trunk = 2

architectures = [([100,80], [100,100,80]),
                 ([100,100,80], [100,80])
                 ([100,100,80], [100,100,80]),
                 ([200,160], [200,160]),
                 ([200,200,160], [200,160])]
n_conv_layers_list = [0,3]

activation_branch = torch.nn.ReLU()
activation_trunk = torch.nn.Sigmoid()

# weights for physics and data loss: loss = w_ph*loss_ph + w_d*loss_d
weight_physics = 0.
weight_data = 1. - weight_physics

# define loss functions (one for CPU evaluation, the other for device (cuda or CPU))
BCs = [BCs[i].to(device) for i in range(4)]


loss_fn_physics_CPU = lambda preds, targets, u, x: torch.tensor(0.)#weight_physics * physics_loss(u, x, preds, [BCs[i].to('cpu') for i in range(4)])
loss_fn_physics_device = lambda preds, targets, u, x: torch.tensor(0.)#weight_physics * physics_loss(u, x, preds, BCs)

loss_fn_data = lambda preds, targets, u, x: weight_data * torch.nn.MSELoss()(preds.view_as(targets), targets)

weight_penalty = 0. # L2 penalty for NN weights
learning_rates = [1e-3]

iterations = 3000 # no. of training epochs


"""
Train the various models
"""
retrain_if_low_r2 = False # retrain model one additional time if R2 on test set is below desired score. The model is discarded and a new one initialised if the retrain still yields R2<0.95.
max_n_retrains = 20 # max. no. of retrains (to avoid potential infinite retrain loop)
desired_r2 = 0.99

for n_conv_layers in n_conv_layers_list:
    print("Using", n_conv_layers, "conv layers")
    for architecture in architectures:
        branch_architecture, trunk_architecture = architecture
        model_params = str(branch_architecture) + "_" + str(trunk_architecture)
        print("Training with branch architecture", branch_architecture, "\nTrunk architecture", trunk_architecture, "\n")
        
        for lr in learning_rates:
            print("Using learning rate", lr, "\n")
            
            models_list = [] # list to store models
            
            # create metrics dict
            metrics = dict(test_loss = [], 
                           R2 = [],
                           training_times=[])
            
            # create dict to store the training loss histories
            loss_histories = dict(total = [], 
                              data = [],
                              physics = [],
                              validation = [])
            
            m = 0
            n_retrains = 0
            while m < n_models:
                m += 1
                print("Training model", str(m) + "/" + str(n_models))
                data = train_data[m-1]
                u_train = (data["u"] + u_shift).log()
                y_train = data["y"]
                r_train = data["r"]
                dataset = DeepONetDataset(u_train, r_train, y_train)
                model = DeepONet(input_size_branch,
                                 input_size_trunk,
                                 branch_architecture,
                                 trunk_architecture,
                                 activation_branch=activation_branch,
                                 activation_trunk=activation_trunk,
                                 use_dropout=False,
                                 n_conv_layers=n_conv_layers,
                                 final_activation_trunk=True)
                model.to(device)
                time_start = time.time()
                
                loss_hist, loss_data_hist, loss_physics_hist, loss_hist_val = train_DON(model, 
                                                                                    dataset,
                                                                                    dataset_val,
                                                                                    iterations, 
                                                                                    loss_fn_data,
                                                                                    loss_fn_physics_device,
                                                                                    batch_size_fun=batch_size_fun,
                                                                                    batch_size_loc=batch_size_loc,
                                                                                    lr=lr,
                                                                                    weight_penalty=weight_penalty,
                                                                                    device=device,
                                                                                    print_every=20)
                
                time_end = time.time()
                training_time = time_end - time_start
                
                model.to("cpu")
                preds = model(u_test, r_test)
                
                r2 = 1. - torch.mean(((preds.flatten(start_dim=1)-y_test.flatten(start_dim=1))**2).mean(axis=1)/y_test.flatten(start_dim=1).var(axis=1))
                if retrain_if_low_r2:
                    if r2 < desired_r2:
                        print("R2 = {:.2f} < {:.2f}, retraining for {:g} epochs.".format(r2, desired_r2, iterations))
                        n_retrains += 1
                        model.to(device)
                        time_start = time.time()
                        loss_hist_new, loss_data_hist_new, loss_physics_hist_new, loss_hist_val_new = train_DON(model, 
                                                                                                            dataset,
                                                                                                            dataset_val,
                                                                                                            iterations, 
                                                                                                            loss_fn_data,
                                                                                                            loss_fn_physics_device,
                                                                                                            batch_size_fun=batch_size_fun,
                                                                                                            batch_size_loc=batch_size_loc,
                                                                                                            lr=lr,
                                                                                                            weight_penalty=weight_penalty,
                                                                                                            device=device,
                                                                                                            print_every=20)
                        time_end = time.time()
                        training_time = training_time + time_end - time_start
                        
                        loss_hist = torch.cat((loss_hist, loss_hist_new))
                        loss_data_hist = torch.cat((loss_data_hist, loss_data_hist_new))
                        loss_physics_hist = torch.cat((loss_physics_hist, loss_physics_hist_new))
                        loss_hist_val = torch.cat((loss_hist_val, loss_hist_val_new))
                        
                        model.to("cpu")
                        preds = model(u_test, r_test)
                        
                        r2 = 1. - torch.mean(((preds.flatten(start_dim=1)-y_test.flatten(start_dim=1))**2).mean(axis=1)/y_test.flatten(start_dim=1).var(axis=1))
                        if r2 < desired_r2:
                            # abandon current model and reinitialise
                            if n_retrains >= max_n_retrains:
                                print("Break training to avoid infinite retraining loop. Adjust training parameters and rerun the code.")
                                print("Trained {:g} models.".format(m))
                                print()
                                break
                            print("Model retraining failed. R2 = {:.2f} < {:.2f}, reinitialising model.".format(r2, desired_r2))
                            print()
                            m -= 1
                            continue
                        else:
                            n_retrains -= 1 # let a successful retraining give more "slack" for later retrainings
                
                # calculate test losses
                test_loss_data = torch.nn.MSELoss()(preds.view_as(y_test), y_test).item()
                test_loss_physics = physics_loss(u_test, r_test, preds, [BCs[i].to('cpu') for i in range(4)])
                test_losses = (test_loss_data, test_loss_physics.item())
                
                metrics["test_loss"].append(test_losses)
                metrics["R2"].append( r2 )
                metrics["training_times"].append(training_time)
                
                models_list.append(model)
                
                loss_histories["total"].append(loss_hist.to('cpu'))
                loss_histories["data"].append(loss_data_hist.to('cpu'))
                loss_histories["physics"].append(loss_physics_hist.to('cpu'))
                loss_histories["validation"].append(loss_hist_val.to('cpu'))
                
                
                
                print()
            
            print(metrics)
            
            # save training_loss
            filename_loss_history = "loss_history_" + str(n_conv_layers) + "_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_loss_history), "wb") as outfile:
                pickle.dump(loss_histories, outfile)
            # save metrics
            filename_metrics = "metrics_" + str(n_conv_layers) + "_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_metrics), "wb") as outfile:
                pickle.dump(metrics, outfile)
            # save models
            filename_models_list = "models_list_" + str(n_conv_layers) + "_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_models_list), "wb") as outfile:
                pickle.dump(models_list, outfile)
            
            print()
print()
print("####################################")
print("#         Training complete.       #")
print("####################################")