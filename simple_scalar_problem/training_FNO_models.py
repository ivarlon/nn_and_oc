# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:42:30 2024

Trains Fourier neural operators
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from FNO import FNO
from utils.training_routines import train_FNO
from CustomDataset import *
from generate_data import generate_data, generate_controls
import time # to measure training time

# seed pytorch RNG
seed = 12345
torch.manual_seed(seed)

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:1")
else:
    print("Using CPU")
    device = torch.device("cpu")

N = 64 # number of points x_i in domain
y0 = 1. # initial condition on state
pf = 0. # terminal condition on adjoint

n_models = 1 # no. of models to train

################################
# Generate train and test data #
################################

n_train = 1000 # no. of training samples
n_test = 100 # no. of test samples
n_val = 100
batch_size = 100 # minibatch size during SGD
basis = "Legendre"
n_coeffs = 6

# generate different training data for different models?
different_data = False
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data(N, basis, n_samples=n_train, n_coeffs=n_coeffs, coeff_range=2., boundary_condition=y0)
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data(N, basis, n_samples=n_train, n_coeffs=n_coeffs, coeff_range=2., boundary_condition=y0)
    train_data = n_models*[data]

# generate test and validation data
test_data = generate_data(N, basis, n_samples=n_test + n_val, coeff_range=2., boundary_condition=y0)
u_test = test_data["u"][:n_test]; x_test = test_data["x"][:n_test]; y_test = test_data["y"][:n_test]
dataset_test = (u_test, y_test)
u_val = test_data["u"][n_test:]; x_val = test_data["x"][n_test:]; y_val = test_data["y"][n_test:]
dataset_val = (u_val, y_val)

# data augmentation: double data set with (e^x u_i, e^x y_i)
#data = torch.cat((data, torch.einsum('i,...i->...i', torch.exp(torch.linspace(0.,1.,N)), data)), axis=0)

training_times = [] # measure training time
loss_test = [] # save test loss for each model




#######################################
# Set up and train the various models #
#######################################

model_name = "FNO"
d_u = 1 # dimension of input u(x_i): u is an Nxd_u array
architectures = torch.cartesian_prod(torch.arange(1,4), torch.tensor([1,4,8])) # pairs of (n_layers, d_v)
architectures = torch.tensor([[3,8]])

loss_fn = torch.nn.MSELoss()

weight_penalties = [1e-3]#, 1e-2, 1e-3]
models_list = []
loss_histories = []

iterations = 4000 # no. of training epochs
lr = 1e-3 # learning rate

for weight_penalty in weight_penalties:
    print("Using weight penalty", weight_penalty)
    for architecture in architectures:#[([N, 200, 50], [1,100,50]), ([N, 300, 100], [1,100,100])]:
        n_layers, d_v = architecture
        model_params = str(n_layers.item()) + "_" + str(d_v.item())
        print("Training with parameters", model_params)
        for m in range(n_models):
            print("Training model", str(m+1) + "/" + str(n_models))
            data = train_data[m]
            u_train = data["u"]
            y_train = data["y"]
            model = FNO(n_layers, N, d_u, d_v)
            dataset = BasicDataset(u_train, y_train)
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
            preds = model(u_test)
            loss_test.append( loss_fn(preds, y_test).item() )
            i = 0
            plt.plot(x_test[i].detach().numpy().ravel(), preds[i].detach().numpy().ravel())
            plt.plot(x_test[i].detach().numpy().ravel(), y_test[i].detach().numpy().ravel(), linestyle="--")
            #torch.save(model, model_name + str(weight_penalty) + "_" + model_params + "_" + str(m+1) + ".pt")
            #np.save(model_name + "_loss_train_" + str(weight_penalty) + "_" + model_params + "_" + str(m+1) + ".npy", loss_history.detach().numpy())
        #np.save(model_name + "_loss_test_" + str(weight_penalty) + "_" + model_params + ".npy", loss_test)
        #np.save(model_name + "_training_time_" + model_params + ".npy", training_time)
print("####################################")
print("#         Training complete.       #")
print("####################################")

"""
y_d = torch.ones(N)
adjoint_data = generate_data(N, n_samples=n_samples, seed=1, generateAdjoint=True, y_d=y_d, boundary_condition=pf)
adjoint_model = FNO(n_layers, N)
y = adjoint_data[:,0]; p = adjoint_data[:,1]
adjoint_loss_history = train(adjoint_model, 
                             y[:,:,None], p[:,:,None],
                             iterations,  
                             train_adjoint=True,
                             weight_prior=0.,
                             boundary_condition=0.)

"""