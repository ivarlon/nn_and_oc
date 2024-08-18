# -*- coding: utf-8 -*-
"""
Trains DeepONet to solve adjoint equation -p' = -p + y-y_d
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
from generate_data import generate_data, augment_data

# seed pytorch RNG
seed = 4321
torch.manual_seed(seed)

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:1")
else:
    print("Using CPU")
    device = torch.device("cpu")

# create data directory to store models and results
data_dir_name = 'adjoint_experiments_DON'
problem_dir_name = "simple_scalar_problem"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, problem_dir_name, data_dir_name)
if not os.path.exists(data_dir):
        os.makedirs(data_dir)

N = 64 # number of points x_i
pf = 0. # terminal condition on adjoint
y_d = 1.5*torch.ones(N) # desired state for OC

n_models = 5

################################
# Generate train and test data #
################################

n_train = 5000 # no. of training samples
n_test = 500 # no. of test samples
n_val = 500 # no. of training samples
batch_size_fun = 200 # minibatch size during SGD
batch_size_loc = N # no. of minibatch domain points. Get worse performance when not using entire domain :/
basis = "Legendre"
n_coeffs = 6

def generate_data_func(n_samples, n_augmented_samples):
    data = generate_data(N, basis, n_samples=n_samples, n_coeffs=n_coeffs, coeff_range=2., boundary_condition=pf, generate_adjoint=True, y_d=y_d)
    augment_data(data, n_augmented_samples=n_augmented_samples, n_combinations=5, max_coeff=2., adjoint=True)
    return data

# generate different data for different models?
different_data = True
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data_func(n_train-2000, 2000)
        data["x"].requires_grad = True
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data_func(n_train-2000, 2000)
    data["x"].requires_grad = True
    train_data = n_models*[data]

# generate test data
test_data = generate_data_func(n_test-200, 200)
test_data["x"].requires_grad = True
y_yd_test = test_data["y-y_d"]; x_test = test_data["x"]; p_test = test_data["p"]

val_data = generate_data_func(n_val-200, 200)
val_data["x"].requires_grad = True
y_yd_val = val_data["y-y_d"]; x_val = val_data["x"]; p_val = val_data["p"]
dataset_val = (y_yd_val.to(device), x_val.to(device), p_val.to(device))


#########################
# Defining physics loss #
#########################

def ODE(y,x,p):
    # y is y-y_d
    dp_x = torch.autograd.grad(outputs=p, inputs=x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    return -dp_x + p - y.view_as(p)

def physics_loss(y, x, p, weight_boundary=1.):
    # y is really y-y_d
    # p = p(x;y,yd) is output of DeepONet
    # x is input tensor and has shape (n_samples, n_domain_points, dim(X))
    ODE_loss = (ODE(y, x, p)**2).mean()
    boundary_loss = torch.nn.MSELoss()(p[:,-1], pf*torch.ones_like(p[:,-1]))
    return ODE_loss + weight_boundary*boundary_loss


#######################################
# Set up and train the various models #
#######################################

input_size_branch = N
input_size_trunk = 1

architectures = [ ( [50,10], [10, 10] ),  ( [100,10], [20, 10] ), ( [100, 20], [20, 20] ), ( [200, 20], [20, 20] ), ( [500, 500, 50], [50, 50] ) ]
#architectures = [([100,10], [20,10])]
n_conv_layers_list = [0,2]

activation_branch = torch.nn.ReLU()
activation_trunk = torch.nn.Sigmoid()

# weights for physics and data loss: loss = w_ph*loss_ph + w_d*loss_d
weight_physics = 0.5
weight_data = 1. - weight_physics

loss_fn_physics = lambda preds, targets, u, x: weight_physics * physics_loss(u,x,preds)
loss_fn_data = lambda preds, targets, u, x: weight_data * torch.nn.MSELoss()(preds, targets)

weight_penalty = 0. # L2 penalty for NN weights

iterations = 5000 # no. of training epochs
learning_rates = [1e-2, 1e-3] # learning rates
# training loop uses Adam optimizer by default

"""
Train the various models
"""

retrain_if_low_r2 = False # retrain model one additional time if R2 on test set is below 95%. The model is discarded and a new one initialised if the retrain still yields R2<0.95.

for n_conv_layers in n_conv_layers_list:
    print("Using", n_conv_layers, "conv layers")    
    for architecture in architectures:
        branch_architecture, trunk_architecture = architecture
        model_params = str(branch_architecture) + "_" + str(trunk_architecture)
        print("Training with branch architecture", branch_architecture, "\nTrunk architecture", trunk_architecture, "\n")
        
        for lr in learning_rates:
            print("Using learning rate", lr)
        
            models_list = [] # list to store models
            
            # create metrics dict
            metrics = dict(test_loss = [], 
                           R2 = [],
                           training_times=[])
            
            # create dict to store the training loss histories
            loss_histories = dict(total = [], 
                              data = [],
                              physics = [])
            m = 0
            n_retrains = 0
            while m < n_models:
                m += 1
                print("Training model", str(m) + "/" + str(n_models))
                data = train_data[m-1]
                y_yd_train = data["y-y_d"]
                p_train = data["p"]
                x_train = data["x"]
                dataset = DeepONetDataset(y_yd_train, x_train, p_train, device=device)
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
                
                loss_history, loss_data_history, loss_physics_history = train_DON(model, 
                                                                            dataset,
                                                                            dataset_val,
                                                                            iterations, 
                                                                            loss_fn_data,
                                                                            loss_fn_physics,
                                                                            batch_size_fun=batch_size_fun,
                                                                            batch_size_loc=batch_size_loc,
                                                                            lr=lr,
                                                                            weight_penalty=weight_penalty)
                time_end = time.time()
                training_time = time_end - time_start
                
                model.to("cpu")
                preds = model(y_yd_test, x_test)
                test_loss_data = torch.nn.MSELoss()(preds, p_test).item()
                test_loss_physics = physics_loss(y_yd_test, x_test, preds).item()
                test_losses = (test_loss_data, test_loss_physics)
                
                r2 = 1. - torch.mean(((preds-p_test)**2).mean(axis=1)/p_test.var(axis=1))
                if retrain_if_low_r2:
                    if r2 < 0.95:
                        print("R2 = {:.2f} < 0.95, retraining for {:g} epochs.".format(r2, iterations))
                        n_retrains += 1
                        model.to(device)
                        time_start = time.time()
                        loss_history_new, loss_data_history_new, loss_physics_history_new = train_DON(model, 
                                                                                    dataset,
                                                                                    dataset_val,
                                                                                    iterations, 
                                                                                    loss_fn_data,
                                                                                    loss_fn_physics,
                                                                                    batch_size_fun=batch_size_fun,
                                                                                    batch_size_loc=batch_size_loc,
                                                                                    lr=lr,
                                                                                    weight_penalty=weight_penalty)
                        time_end = time.time()
                        training_time = training_time + time_end - time_start
                        
                        loss_history = torch.cat((loss_history, loss_history_new))
                        loss_data_history = torch.cat((loss_data_history, loss_data_history_new))
                        loss_physics_history = torch.cat((loss_physics_history, loss_physics_history_new))
                        
                        model.to("cpu")
                        preds = model(y_yd_test, x_test)
                        test_loss_data = torch.nn.MSELoss()(preds, p_test).item()
                        test_loss_physics = physics_loss(y_yd_test, x_test, preds).item()
                        test_losses = (test_loss_data, test_loss_physics)
                        
                        r2 = 1. - torch.mean(((preds-p_test)**2).mean(axis=1)/p_test.var(axis=1))
                        
                        if r2 <0.95:
                            # abandon current model and reinitialise
                            assert n_retrains <= 10, "Break training to avoid infinite retraining loop. Adjust training parameters and rerun the code."
                            print("Model retraining failed. R2 = {:.2f} < 0.95, reinitialising model.".format(r2))
                            print()
                            m -= 1
                            continue
                
                
                metrics["test_loss"].append(test_losses)
                metrics["R2"].append(r2)
                print("R2 = {:.2f}".format(r2))
                
                metrics["training_times"].append(round(training_time,1))
                
                models_list.append(model)
                
                loss_histories["total"].append(loss_history.to('cpu'))
                loss_histories["data"].append(loss_data_history.to('cpu'))
                loss_histories["physics"].append(loss_physics_history.to('cpu'))
                
                
                
            print()
            
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