# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:42:30 2024

Trains neural operators

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
from CustomDataset import *
#train, training_loop = training_routines.train, training_routines.training_loop
from generate_data import generate_data, normalise_tensor
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
    architectures = torch.tensor([[3,8]])
    model_name = "FNO"
elif useDeepONet:
    trunk_architectures = [[100, 50], [200, 100]]
    branch_architectures = [ [200, 50], [200, 200, 50] ]
    input_size_branch = N
    input_size_trunk = 1
    architectures = [ ([1000, 800, 100], [200, 100, 100]) ]
    activation_branch = torch.nn.ReLU()
    activation_trunk = torch.nn.Sigmoid()
    model_name = "DeepONet"
    x = torch.linspace(0.,1.,N).view(N,-1) # define the domain for the trunk net
weight_penalties = [0.]#, 1e-2, 1e-3]
n_models = 1

models_list = []
loss_histories = []

iterations = 500 # no. of training epochs

"""
Generate train and test data
"""
n_samples = 1000 # total no. of samples
n_test = int(0.15*n_samples) # no. of test samples
n_train = n_samples - n_test # no. of training samples
batch_size = 100 # minibatch size during SGD
basis = "Bernstein"
# generate different data for different models?
different_data = True
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data(N, basis, n_samples=n_train, coeff_range=2., boundary_condition=y0)
        train_data.append(data)
else:
    data = generate_data(N, basis, n_samples=n_train, coeff_range=2., boundary_condition=y0)
    train_data = n_models*[data]

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
for weight_penalty in [0.]:#weight_penalties:
    print("Using weight penalty", weight_penalty)
    for architecture in architectures:#[([N, 200, 50], [1,100,50]), ([N, 300, 100], [1,100,100])]:
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
            data = train_data[m]
            u_normalised = normalise_tensor(data[:,0], dim=1).unsqueeze(-1)
            y_train = data[:,1,:,None]
            if useFNO:
                model = FNO(n_layers, N, d_u, d_v)
                dataset = BasicDataset(u_normalised, y_train)
                dataloader = DataLoader(dataset, batch_size=batch_size)
            else:
                model = DeepONet(input_size_branch,
                                 input_size_trunk,
                                 branch_architecture,
                                 trunk_architecture,
                                 activation_branch=activation_branch,
                                 activation_trunk=activation_trunk)
                dataset = DeepONetDataset(u_normalised, x, y_train)
                dataloader = DeepONetDataloader(dataset, fun_batch_size=batch_size, loc_batch_size=N//2)
            
            model.to(device)
            time_start = time.time()
            loss_history = train(model, 
                                dataloader, 
                                iterations, 
                                boundary_condition=y0,
                                weight_penalty=weight_penalty)
            time_end = time.time()
            training_time = round(time_end-time_start,1)
            model.to('cpu')
            models_list.append(model)
            loss_histories.append(loss_history.to('cpu'))
            if useFNO:
                preds = model(normalise_tensor(u_test[:,:,None], dim=1))
            else:
                preds = model(normalise_tensor(u_test[:,:,None], dim=1), x.repeat(n_test,1,1))
            loss_test.append( torch.mean((preds.flatten() - y_test.flatten())**2).item() )
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