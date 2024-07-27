# -*- coding: utf-8 -*-
"""
Trains Fourier neural operators to solve adjoint eq. in control of heat eq.
"""

# for saving data
import sys
import os
import pickle
import time # to measure training time

# import numberical libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# import custom libraries and functions
from FNO import FNO
from training_routines import train_FNO
from CustomDataset import *
from generate_data_heat_eq import generate_data, augment_data

# seed pytorch RNG
seed = 123
torch.manual_seed(seed)


if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:1")
else:
    print("Using CPU")
    device = torch.device("cpu")

# create data directory to store models and results
data_dir_name = 'adjoint_experiments_FNO'
problem_dir_name = "heat_equation"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, problem_dir_name, data_dir_name)
if not os.path.exists(data_dir):
        os.makedirs(data_dir)


diffusion_coeff = 0.25 # coefficient multiplying curvature term y_xx

N_t = 16 # number of time points t_i
N_x = 8 # number of spatial points x_j

# time span
T = 0.2
t0 = 0.; tf = t0 + T

# domain length
L = 0.1
x0 = 0.; xf = x0 + L

# boundary conditions
p_TC = 2.*torch.sin(torch.linspace(0., 2*np.pi, N_x))**2 # terminal condition on adjoint is zero
p_BCs = (torch.zeros(N_t), torch.zeros(N_t)) # zero Dirichlet boundary conditions

y_d = 1.5*torch.sin(torch.meshgrid(torch.linspace(0., np.pi, N_t), torch.zeros(N_x))[0])**10 # desired state for OC is single peak

n_models = 1 # number of models to train


################################
# Generate train and test data #
################################

n_train = 1000 # no. of training samples
n_test = 100 # no. of test samples
n_val = 100
batch_size = 100 # minibatch size during SGD

n_t_coeffs = 4
n_x_coeffs = 5
u_max = 10. # maximum amplitude of control

generate_data_func = lambda n_samples: generate_data(N_t, N_x, t_span=(t0,tf), x_span=(x0,xf),
                  IC=p_TC,
                  BCs=p_BCs,
                  n_t_coeffs=n_t_coeffs,
                  n_x_coeffs=n_x_coeffs,
                  n_samples=n_samples,
                  u_max=u_max,
                  diffusion_coeff=diffusion_coeff,
                  y_d=y_d.numpy(),
                  generate_adjoint=True)

# define function to reshape matrix tensors y(t_i,x_j) into flat vectors with singleton final dim
# so that the FNO can process them. Tensors need to be reshaped into matrices afterwards.
flatten_tensors = lambda tens: tens.flatten(start_dim=1).unsqueeze(-1)
reshape_tensors = lambda tens: tens[...,0].view(tens.shape[0], N_t, N_x)

# generate different training data for different models?
different_data = True
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data_func(n_train)
        train_data.append(data)
else:
    # use the same training data for all models
    data = data = generate_data_func(n_train)
    train_data = n_models*[data]

# generate test and validation data
test_data = generate_data_func(200)
#augment_data(test_data, n_augmented_samples=n_test-200, n_combinations=5, max_coeff=2)
y_y_d_test = flatten_tensors(test_data["y-y_d"]); p_test = flatten_tensors(test_data["p"])

val_data = generate_data_func(200)
#augment_data(val_data, n_augmented_samples=n_val-200, n_combinations=5, max_coeff=2)
dataset_val = (flatten_tensors(val_data["y-y_d"]).to(device), flatten_tensors(test_data["p"]).to(device))


training_times = [] # measure training time
loss_test = [] # save test loss for each model




#######################################
# Set up and train the various models #
#######################################

model_name = "FNO"
d_u = 1 # dimension of input y(t_i,x_j)
architectures = torch.cartesian_prod(torch.arange(1,4), torch.tensor([1,4,8])) # pairs of (n_layers, d_v)
architectures = torch.tensor([[3,8]])

loss_fn = torch.nn.MSELoss()

weight_penalties = [0]#, 1e-2, 1e-3]
models_list = []
loss_histories = []

iterations = 1000 # no. of training epochs
lr = 1e-2 # learning rate

for weight_penalty in weight_penalties:
    print("Using weight penalty", weight_penalty)
    for architecture in architectures:#[([N, 200, 50], [1,100,50]), ([N, 300, 100], [1,100,100])]:
        n_layers, d_v = architecture
        model_params = str(n_layers.item()) + "_" + str(d_v.item())
        print("Training with parameters", model_params)
        for m in range(n_models):
            print("Training model", str(m+1) + "/" + str(n_models))
            data = train_data[m]
            y_y_d_train = flatten_tensors(data["y-y_d"])
            p_train = flatten_tensors(data["p"])
            model = FNO(n_layers, N_t*N_x, d_u, d_v)
            dataset = BasicDataset(y_y_d_train, p_train, device=device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.to(device)
            time_start = time.time()
            loss_history = train_FNO(model,
                                dataloader, 
                                dataset_val,
                                iterations, 
                                loss_fn,
                                lr=lr,
                                weight_penalty=weight_penalty)
            time_end = time.time()
            training_time = round(time_end-time_start,1)
            model.to('cpu')
            models_list.append(model)
            loss_histories.append(loss_history.to('cpu'))
            preds = model(y_y_d_test)
            loss_test.append( loss_fn(preds, p_test).item() )
            """
            # save training_loss
            filename_loss_history = "loss_history_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_loss_history), "wb") as outfile:
                pickle.dump(loss_histories, outfile)
            # save test loss
            filename_test_loss = "test_loss_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_test_loss), "wb") as outfile:
                pickle.dump(loss_test, outfile)
            # save models
            filename_models_list = "models_list_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_models_list), "wb") as outfile:
                pickle.dump(models_list, outfile)
            # save training time
            filename_training_times = "training_times_" + model_params + "_" + str(lr) + ".pkl"
            with open(os.path.join(data_dir, filename_training_times), "wb") as outfile:
                pickle.dump(training_times, outfile)
            print()"""
print()
print("####################################")
print("#         Training complete.       #")
print("####################################")
