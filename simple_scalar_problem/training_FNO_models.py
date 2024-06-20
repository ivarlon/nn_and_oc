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
from FNO import FNO, train
from generate_data import generate_data
import time # to measure training time

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:1")
else:
    print("Using CPU")
    device = torch.device("cpu")

N = 64 # number of points x_i
d_u = 1 # dimension of input u(x_i): u is an Nxd_u array
#d_v = 8 # lifted dimension (determines size of Fourier kernel: for each wavenumber a d_v*d_v matrix)

y0 = 1. # initial condition on state
pN = 0. # terminal condition on adjoint

architectures = torch.cartesian_prod(torch.arange(1,4), torch.tensor([1,4,8]))
#architectures = torch.tensor([[3,8]])
weight_penalties = [0., 1e-2, 1e-3]
n_models = 5

models_list = []
loss_histories = []

iterations = 2000 # no. of training epochs
n_samples = 1000 # total no. of samples
n_test = int(0.15*n_samples) # no. of test samples
n_train = n_samples - n_test # no. of training samples
basis = "Bernstein"
train_data = []
# generate different data for different models
for m in range(n_models):
    data = generate_data(N, basis, n_samples=n_train, seed=m, coeff_range=2., boundary_condition=y0)
    train_data.append(data)

# generate test data
test_data = generate_data(N, basis, n_samples=n_test, seed=420, coeff_range=2., boundary_condition=y0)
u_test = test_data[:,0]; y_test = test_data[:,1]
# data augmentation: double data set with (e^x u_i, e^x y_i)
#data = torch.cat((data, torch.einsum('i,...i->...i', torch.exp(torch.linspace(0.,1.,N)), data)), axis=0)
# scramble data
#data = data[torch.randperm(2*n_sample)]

for weight_penalty in weight_penalties:
    print("Using weight penalty", weight_penalty)
    for architecture in architectures:
        n_layers, d_v = architecture
        print("Training with parameters", n_layers.item(), d_v.item())
        training_times = [] # measure training time
        loss_test = [] # save test loss for each model
        for m in range(n_models):
            print("Training model", str(m+1) + "/" + str(n_models))
            model = FNO(n_layers, N, d_u, d_v)
            model.to(device)
            data = train_data[m].to(device)
            u = data[:,0]; y = data[:,1]
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
            torch.save(model, "FNO_" + str(weight_penalty) + "_" + str(n_layers.item()) + "_" + str(d_v.item()) + "_" + str(m+1) + ".pt")
            np.save("FNO_loss_train_" + str(weight_penalty) + "_" + str(n_layers.item()) + "_" + str(d_v.item()) + "_" + str(m+1) + ".npy", loss_history.detach().numpy())
        np.save("FNO_loss_test_" + str(weight_penalty) + "_" + str(n_layers.item()) + "_" + str(d_v.item()) + ".npy", loss_test)
        np.save("FNO_training_time_" + str(n_layers.item()) + "_" + str(d_v.item()) + ".npy", training_time)
print("####################################")
print("#         Training complete.       #")
print("####################################")

"""
y_d = torch.ones(N)
adjoint_data = generate_data(N, n_samples=n_samples, seed=1, generateAdjoint=True, y_d=y_d, boundary_condition=pN)
adjoint_model = FNO(n_layers, N)
y = adjoint_data[:,0]; p = adjoint_data[:,1]
adjoint_loss_history = train(adjoint_model, 
                             y[:,:,None], p[:,:,None],
                             iterations,  
                             train_adjoint=True,
                             weight_prior=0.,
                             boundary_condition=0.)

"""