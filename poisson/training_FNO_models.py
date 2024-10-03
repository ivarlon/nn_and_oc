# -*- coding: utf-8 -*-
"""
Trains Fourier neural operators (FNOs) to solve Poisson equation

"""

# for saving data
import sys
import os
import pickle
import time # to measure training time

# import numerical libraries
import numpy as np
import torch
from torch.utils.data import DataLoader

# import custom libraries and functions
from FNO import FNO
from utils.training_routines import train_FNO
from CustomDataset import *
from generate_data_poisson import generate_data, augment_data

# seed pytorch RNG
seed = 1789
torch.manual_seed(seed)

cuda = 2 # 1,2,3

if torch.cuda.is_available():
    print("Using CUDA", cuda)
    device = torch.device("cuda:{}".format(cuda))
else:
    print("Using CPU\n")
    device = torch.device("cpu")

# create data directory to store models and results
data_dir_name = 'state_experiments_FNO'
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
batch_size = 50 # minibatch size during SGD

u_max = 80. # maximum amplitude of control

generate_data_func = lambda n_samples: generate_data(N,
                  L,
                  BCs=[bc.numpy() for bc in BCs],
                  n_samples=n_samples,
                  u_max=u_max)

# define function to reshape matrix tensors u(t_i,x_j) into flat vectors with singleton final dim
# so that the FNO can process them. Tensors need to be reshaped into matrices afterwards.
flatten_tensors = lambda tens: tens.flatten(start_dim=1).unsqueeze(-1)
reshape_tensors = lambda tens: tens[...,0].view(tens.shape[0], N, N)

# generate different data for different models?
different_data = True
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data_func(n_train-2000)
        augment_data(data, n_augmented_samples=2000, n_combinations=5, max_coeff=2)
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data_func(n_train-2000)
    augment_data(data, n_augmented_samples=2000, n_combinations=5, max_coeff=2)
    train_data = n_models*[data]

# generate test and validation data
test_data = generate_data_func(n_test-200)
augment_data(test_data, n_augmented_samples=200, n_combinations=5, max_coeff=2)
u_test = flatten_tensors(test_data["u"]); y_test = flatten_tensors(test_data["y"])

val_data = generate_data_func(n_val-200)
augment_data(val_data, n_augmented_samples=200, n_combinations=5, max_coeff=2)
dataset_val = (flatten_tensors(val_data["u"]).to(device), flatten_tensors(val_data["y"]).to(device))

#######################################
# Set up and train the various models #
#######################################

d_u = 1 # dimension of input u(x_i): u is an Nxd_u array
architectures = torch.cartesian_prod(torch.arange(3,6), torch.tensor([4,8,16,32])) # pairs of (n_layers, d_v)

loss_fn = torch.nn.MSELoss()

weight_penalties = [0]#, 1e-2, 1e-3]
learning_rates = [1e-3] # learning rate

iterations = 3000 # no. of training epochs

"""
Train the models
"""
retrain_if_low_r2 = False # retrain model one additional time if R2 on test set is below desired score. The model is discarded and a new one initialised if the retrain still yields R2<0.95.
max_n_retrains = 20 # max. no. of retrains (to avoid potential infinite retrain loop)
desired_r2 = 0.99

for weight_penalty in weight_penalties:
    print("Using weight penalty", weight_penalty)
    
    for architecture in architectures:
        n_layers, d_v = architecture
        model_params = str(n_layers.item()) + "_" + str(d_v.item())
        print("Training with parameters", model_params)
        for lr in learning_rates:
            models_list = [] # list to store models
            
            # create metrics dict
            metrics = dict(test_loss = [], 
                           R2 = [],
                           training_times=[])
            
            loss_histories = dict(train = [],
                                  validation = []) # dict to store loss histories
            
            m = 0
            n_retrains = 0
            while m < n_models:
                m += 1
                print("Training model", str(m) + "/" + str(n_models))
                data = train_data[m-1]
                u_train = flatten_tensors(data["u"])
                y_train = flatten_tensors(data["y"])
                dataset = BasicDataset(u_train, y_train)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                model = FNO(n_layers, N*N, d_u, d_v)
                model.to(device)
                time_start = time.time()
                loss_hist, loss_hist_val = train_FNO(model,
                                                    dataloader, 
                                                    dataset_val,
                                                    iterations, 
                                                    loss_fn,
                                                    lr=lr,
                                                    weight_penalty=weight_penalty,
                                                    device=device,
                                                    print_every=20)
                time_end = time.time()
                training_time = time_end - time_start
                
                model.to('cpu')
                preds = model(u_test)
                
                r2 = 1. - torch.mean(((preds.flatten(start_dim=1)-y_test.flatten(start_dim=1))**2).mean(axis=1)/y_test.flatten(start_dim=1).var(axis=1)).detach()
                if retrain_if_low_r2:
                    if r2 < desired_r2:
                        print("R2 = {:.2f} < {:.2f}, retraining for {:g} epochs.".format(r2, desired_r2, iterations))
                        n_retrains += 1
                        model.to(device)
                        time_start = time.time()
                        loss_hist_new, loss_hist_val_new = train_FNO(model,
                                                            dataloader, 
                                                            dataset_val,
                                                            iterations, 
                                                            loss_fn,
                                                            lr=lr,
                                                            weight_penalty=weight_penalty,
                                                            device=device,
                                                            print_every=20)
                        time_end = time.time()
                        training_time = training_time + time_end - time_start
                        
                        loss_hist = torch.cat((loss_hist, loss_hist_new))
                        loss_hist_val = torch.cat((loss_hist_val, loss_hist_val_new))
                        model.to('cpu')
                        preds = model(u_test)
                        
                        r2 = 1. - torch.mean(((preds.flatten(start_dim=1)-y_test.flatten(start_dim=1))**2).mean(axis=1)/y_test.flatten(start_dim=1).var(axis=1)).detach()
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
                
                # calculate test loss
                test_loss = loss_fn(preds, y_test).item()
                metrics["test_loss"].append(test_loss)
                metrics["R2"].append( r2 )
                metrics["training_times"].append(training_time)
                
                models_list.append(model)
                
                loss_histories["train"].append(loss_hist.to('cpu'))
                loss_histories["validation"].append(loss_hist_val.to('cpu'))
                
                print()
                
            # save training_loss
            filename_loss_history = "loss_history_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_loss_history), "wb") as outfile:
                pickle.dump(loss_histories, outfile)
            # save metrics
            filename_metrics = "metrics_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_metrics), "wb") as outfile:
                pickle.dump(metrics, outfile)
            # save models
            filename_models_list = "models_list_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_models_list), "wb") as outfile:
                pickle.dump(models_list, outfile)
            
            print()
print()
print("####################################")
print("#         Training complete.       #")
print("####################################")