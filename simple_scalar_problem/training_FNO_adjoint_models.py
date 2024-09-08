# -*- coding: utf-8 -*-
"""
Trains Fourier neural operators to solve adjoint eq. in OC of exp decay eq.
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
import matplotlib.pyplot as plt

# import custom libraries and functions
from FNO import FNO
from utils.training_routines import train_FNO
from CustomDataset import *
from generate_data import generate_data, augment_data

# seed pytorch RNG
seed = 54321
torch.manual_seed(seed)

cuda = 1

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:{}".format(cuda))
else:
    print("Using CPU")
    device = torch.device("cpu")

# create data directory to store models and results
data_dir_name = 'adjoint_experiments_FNO'
problem_dir_name = "simple_scalar_problem"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, problem_dir_name, data_dir_name)
if not os.path.exists(data_dir):
        os.makedirs(data_dir)

N = 64 # number of points x_i in domain
pf = 0. # terminal condition on adjoint
y_d = 1.5*torch.ones(N) # desired state

n_models = 3 # no. of models to train

################################
# Generate train and test data #
################################

n_train = 5000 # no. of training samples
n_test = 500 # no. of test samples
n_val = 500
batch_size = 50 # minibatch size during SGD
basis = "Legendre"
n_coeffs = 6

def generate_data_func(n_samples, n_augmented_samples):
    data = generate_data(N, basis, n_samples=n_samples, n_coeffs=n_coeffs, coeff_range=2., boundary_condition=pf, generate_adjoint=True, y_d=y_d)
    augment_data(data, n_augmented_samples=n_augmented_samples, n_combinations=5, max_coeff=2., adjoint=True)
    return data

# generate different training data for different models?
different_data = True
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data_func(n_train-2000, 2000)
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data_func(n_train-2000, 2000)
    train_data = n_models*[data]

# generate test and validation data
test_data = generate_data_func(n_test-200, 200)
y_yd_test = test_data["y-y_d"].unsqueeze(-1); p_test = test_data["p"]

val_data = generate_data_func(n_val-200, 200)

dataset_val = (val_data["y-y_d"].unsqueeze(-1).to(device), val_data["p"].to(device))


#######################################
# Set up and train the various models #
#######################################

model_name = "FNO"
d_u = 1 # dimension of input function
architectures = torch.cartesian_prod(torch.arange(1,5), torch.tensor([2,4,8,16])) # pairs of (n_layers, d_v)
#architectures = torch.tensor([[3,8]])

def loss_fn(preds, targets, weight_BC=1.):
    # loss is weighted sum of MSE + extra boundary loss to ensure model learns BC
    terminal_cond_loss = weight_BC * ((preds[:,-1] - pf)**2).mean()
    return torch.nn.MSELoss()(preds, targets) + weight_BC*terminal_cond_loss

weight_penalties = [0.]

iterations = 3000 # no. of training epochs
learning_rates = [1e-3] # learning rate

"""
Training models
"""
retrain_if_low_r2 = False # retrain model one additional time if R2 on test set is below desired score. The model is discarded and a new one initialised if the retrain still yields R2<0.95.
max_n_retrains = 20 # max. no. of retrains (to avoid potential infinite retrain loop)
desired_r2 = 0.95

for weight_penalty in weight_penalties:
    print("Using weight penalty", weight_penalty)
    for architecture in architectures:
        n_layers, d_v = architecture
        model_params = str(n_layers.item()) + "_" + str(d_v.item())
        print("Training with parameters", model_params)
        
        for lr in learning_rates:
            print("Using learning rate", lr)
        
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
                y_yd_train = data["y-y_d"].unsqueeze(-1)
                p_train = data["p"]
                model = FNO(n_layers, N, d_u, d_v)
                dataset = BasicDataset(y_yd_train, p_train)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
                                                print_every=50)
                time_end = time.time()
                training_time = time_end - time_start
                
                model.to('cpu')
                preds = model(y_yd_test)
                
                r2 = 1. - torch.mean(((preds-p_test)**2).mean(axis=1)/p_test.var(axis=1))
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
                                                                    print_every=50)
                        
                        time_end = time.time()
                        training_time = training_time + time_end - time_start
                        
                        loss_hist = torch.cat((loss_hist, loss_hist_new))
                        loss_hist_val = torch.cat((loss_hist_val, loss_hist_val_new))
                        
                        model.to('cpu')
                        preds = model(y_yd_test)
                        
                        r2 = 1. - torch.mean(((preds-p_test)**2).mean(axis=1)/p_test.var(axis=1))
                        if r2 < desired_r2:
                            # abandon current model and reinitialise
                            if n_retrains >= max_n_retrains:
                                print("Break training to avoid infinite retraining loop. Adjust training parameters and rerun the code.")
                                print("Trained {:g} models.".format(m-1))
                                print()
                                break
                            print("Model retraining failed. R2 = {:.2f} < 0.95, reinitialising model.".format(r2))
                            print()
                            m -= 1
                            continue
                        else:
                            n_retrains -= 1 # let a successful retraining give more "slack" for later retrainings
                
                test_loss = loss_fn(preds, p_test).item()
                metrics["test_loss"].append(test_loss)
                metrics["R2"].append(r2)
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
