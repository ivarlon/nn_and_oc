# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:42:30 2024

Trains Fourier neural operators

"""
"""
data = generate_data(N, n_samples=500, seed=0)
u = data[:,0]; y = data[:,1]

models = []
loss_histories = []

prior_weights = [0., 1e-3, 1e-2, 1e-1, 1., 10.]
for weight_prior in prior_weights:
    model = FNO(n_layers, N)
    loss_history = train(model, u[:,:,None], y[:,:,None], 1000, weight_prior=weight_prior)
    models.append(model)
    loss_histories.append(loss_history)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from FNO import FNO
from DeepONet import DeepONet
from utils.training_routines import *
from CustomDataSet import FunctionDataSet
#train, training_loop = training_routines.train, training_routines.training_loop
from generate_data import generate_data
import time # to measure training time

# seed pytorch RNG
seed = 12345
torch.manual_seed(seed)

useDeepONet = True
useFNO = not useDeepONet

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:1")
else:
    print("Using CPU")
    device = torch.device("cpu")

N = 64 # number of points x_i
y0 = 1. # initial condition on state
pf = 0. # terminal condition on adjoint

if useFNO:
    d_u = 1 # dimension of input u(x_i): u is an Nxd_u array
    #d_v = 8 # lifted dimension (determines size of Fourier kernel: for each wavenumber a d_v*d_v matrix)
    
    architectures = torch.cartesian_prod(torch.arange(1,4), torch.tensor([1,4,8]))
    #architectures = torch.tensor([[3,8]])
    model_name = "FNO"
elif useDeepONet:
    trunk_architectures = [[100, 50], [200, 100]]
    branch_architectures = [ [200, 50], [200, 200, 50], [] ]
    architectures = [ ([50,50], [200,50]) ]
    model_name = "DeepONet"
    x = torch.linspace(0.,1.,N).view(N,-1) # define the domain for the trunk net
weight_penalties = [0.]#, 1e-2, 1e-3]
n_models = 1

models_list = []
loss_histories = []

iterations = 2000 # no. of training epochs

"""
Generate train and test data
"""
n_samples = 1000 # total no. of samples
n_test = int(0.15*n_samples) # no. of test samples
n_train = n_samples - n_test # no. of training samples
basis = "Bernstein"
train_data = []
# generate different data for different models
for m in range(n_models):
    data = generate_data(N, basis, n_samples=n_train, coeff_range=2., boundary_condition=y0)
    train_data.append(data)

# generate test data
test_data = generate_data(N, basis, n_samples=n_test, coeff_range=2., boundary_condition=y0)
u_test = test_data[:,0]; y_test = test_data[:,1]

# data augmentation: double data set with (e^x u_i, e^x y_i)
#data = torch.cat((data, torch.einsum('i,...i->...i', torch.exp(torch.linspace(0.,1.,N)), data)), axis=0)
# scramble data
#data = data[torch.randperm(2*n_sample)]

"""
Train the various models
"""
for weight_penalty in weight_penalties:
    print("Using weight penalty", weight_penalty)
    for architecture in architectures:
        if useFNO:
            n_layers, d_v = architecture
            model_params = str(n_layers.item()) + "_" + str(d_v.item())
            print("Training with parameters", model_params)
        elif useDeepONet:
            branch_architecture, trunk_architecture = architecture
            model_params = str(branch_architecture) + "_" + str(trunk_architecture)
            print("Training with branch architecture", branch_architecture, "\nTrunk architecture", trunk_architecture)
        training_times = [] # measure training time
        loss_test = [] # save test loss for each model
        for m in range(n_models):
            print("Training model", str(m+1) + "/" + str(n_models))
            if useFNO:
                model = FNO(n_layers, N, d_u, d_v)
                data = train_data[m]
                dataset = FunctionDataSet(data[:,0], data[:,1])
                dataset = dataset.to(device)
                dataloader = DataLoader(dataset, batch_size=batch_size)
            else:
                model = DeepONet(branch_architecture, trunk_architecture)
                # DeepONet is trained on (u, x)
                data = train_data[m]
                x_batch_size = N//2 # basically do SGD with half of domain as a batch
                input_pairs = torch.cartesian_prod((data[:,0], x))
                x_permutation = torch.random
                dataset = FunctionDataSet(input_pairs.flatten(), data[:,1])
            
            model.to(device)
            y = data[:,1]
            time_start = time.time()
            loss_history = train(model, 
                                u[:,:,None], 
                                y[:,:,None], 
                                iterations, 
                                boundary_condition=y0,
                                weight_penalty=weight_penalty)
            time_end = time.time()
            training_time = round(time_end-time_start,1)
            model.to('cpu')
            models_list.append(model)
            loss_histories.append(loss_history.to('cpu'))
            preds = model(u_test[:,:,None])
            loss_test.append( torch.mean((preds.ravel() - y_test.ravel())**2).item() )
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