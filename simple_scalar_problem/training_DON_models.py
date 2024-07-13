# -*- coding: utf-8 -*-
"""
Trains Deep Operator Networks (DeepONets)
"""


import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
import pickle
import numpy as np
import torch
from DeepONet import DeepONet
from utils.training_routines import train_DON
from CustomDataset import *
from generate_data import generate_data, generate_controls
import time # to measure training time

# seed pytorch RNG
seed = 1234
torch.manual_seed(seed)

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:1")
else:
    print("Using CPU")
    device = torch.device("cpu")

train_adjoint = False # either train NN to solve adjoint or to solve state
if train_adjoint:
    def ODE_interior(y,x,p):
        dp_x = torch.autograd.grad(outputs=p, inputs=x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        return -dp_x + p - y.view_as(p)
else:
    def ODE_interior(u,x,y):
        dy_x = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        return dy_x + y - u.view_as(y)

def physics_loss(u, x, y, weight_boundary=1.):
    # y = y(x;u) is output of DeepONet, tensor of shape (n_samples, n_domain_points, dim(Y))
    # x is input tensor and has shape (n_samples, n_domain_points, dim(X))
    interior_loss = (ODE_interior(u, x, y)**2).mean()
    boundary_loss = torch.nn.MSELoss()(y[:,0], y0*torch.ones_like(y[:,0]))
    return interior_loss + weight_boundary*boundary_loss


N = 64 # number of points x_i
y0 = 1. # initial condition on state
pf = 0. # terminal condition on adjoint
y_d = 1.5*torch.ones(N,1) # desired state for OC

n_models = 5

################################
# Generate train and test data #
################################

n_train = 1000 # no. of training samples
n_test = 100 # no. of test samples
n_val = 100 # no. of training samples
batch_size_fun = 50 # minibatch size during SGD
batch_size_loc = N # no. of minibatch domain points. Get worse performance when not using entire domain :/
basis = "Legendre"
n_coeffs = 6

# generate different data for different models?
different_data = False
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data(N, basis, n_samples=n_train, n_coeffs=n_coeffs, coeff_range=2., boundary_condition=y0)
        data["x"].requires_grad = True
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data(N, basis, n_samples=n_train, n_coeffs=n_coeffs, coeff_range=2., boundary_condition=y0)
    data["x"].requires_grad = True
    train_data = n_models*[data]

# generate test data
test_data = generate_data(N, basis, n_samples=n_test + n_val, coeff_range=2., boundary_condition=y0)
test_data["x"].requires_grad = True
u_test = test_data["u"][:n_test]; x_test = test_data["x"][:n_test]; y_test = test_data["y"][:n_test]
dataset_test = (u_test, x_test, y_test)
u_val = test_data["u"][n_test:]; x_val = test_data["x"][n_test:]; y_val = test_data["y"][n_test:]
dataset_val = (u_val.to(device), x_val.to(device), y_val.to(device))

# data augmentation: double data set with (e^x u_i, e^x y_i)
#data = torch.cat((data, torch.einsum('i,...i->...i', torch.exp(torch.linspace(0.,1.,N)), data)), axis=0)



#######################################
# Set up and train the various models #
#######################################

model_name = "DeepONet"
input_size_branch = N
input_size_trunk = 1

final_layer_size1 = 10
final_layer_size2 = 40
branch_hidden_sizes = [100, 200]
trunk_hidden_sizes = [5, 10, 50]

architectures = [ [[branch_hidden_sizes[0],final_layer_size1], [trunk_hidden, final_layer_size1]] for trunk_hidden in trunk_hidden_sizes[:-1]] \
    + [ [ [branch_hidden,final_layer_size2], [trunk_hidden_sizes[-1],final_layer_size2] ] for branch_hidden in branch_hidden_sizes[1:]] \
        + [ [[500, 500, 50], [50, 50]] ]
#architectures = [([100,10], [10,10])]
activation_branch = torch.nn.ReLU()
activation_trunk = torch.nn.Sigmoid()

# weights for physics and data loss: loss = w_ph*loss_ph + w_d*loss_d
weight_physics = 0.5
weight_data = 1. - weight_physics


def loss_fn(preds, targets, u, x):
    loss_physics = physics_loss(u,x,preds)
    loss_data = torch.nn.MSELoss()(preds, targets)
    return weight_physics*loss_physics + weight_data*loss_data

weight_penalties = [0., 1e-2] # L2 penalty for NN weights


iterations = 3000 # no. of training epochs


"""
Train the various models
"""
for weight_penalty in weight_penalties:
    print("Using weight penalty", weight_penalty, "\n")
    for architecture in architectures:
        training_times = [] # measure training time
        test_loss = [] # save test loss for each model
        models_list = []
        loss_histories = []
        branch_architecture, trunk_architecture = architecture
        model_params = str(branch_architecture) + "_" + str(trunk_architecture)
        print("Training with branch architecture", branch_architecture, "\nTrunk architecture", trunk_architecture, "\n")
        for m in range(n_models):
            print("Training model", str(m+1) + "/" + str(n_models))
            data = train_data[m]
            u_train = data["u"]
            y_train = data["y"]
            x_train = data["x"]
            dataset = DeepONetDataset(u_train, x_train, y_train, device=device)
            model = DeepONet(input_size_branch,
                             input_size_trunk,
                             branch_architecture,
                             trunk_architecture,
                             activation_branch=activation_branch,
                             activation_trunk=activation_trunk,
                             use_dropout=False,
                             final_activation_trunk=True)
            model.to(device)
            time_start = time.time()
            
            loss_history = train_DON(model, 
                                dataset,
                                dataset_val,
                                iterations, 
                                loss_fn,
                                batch_size_fun=batch_size_fun,
                                batch_size_loc=batch_size_loc,
                                lr=1e-2,
                                weight_penalty=weight_penalty)
            time_end = time.time()
            training_time = round(time_end-time_start,1)
            model.to('cpu')
            models_list.append(model)
            loss_histories.append(loss_history.to('cpu'))
            preds = model(u_test, x_test)
            test_loss.append(loss_fn(preds, y_test, u_test, x_test).item())
            print()
        # save training_loss
        filename_loss_history = "loss_history_" + model_params + "_" + str(weight_penalty) + "_" + model_name + ".pkl"
        with open(filename_loss_history, "wb") as outfile:
            pickle.dump(loss_histories, outfile)
        # save test loss
        filename_test_loss = "test_loss_" + model_params + "_" + str(weight_penalty) +  "_" + ".pkl"
        with open(filename_test_loss, "wb") as outfile:
            pickle.dump(test_loss, outfile)
        # save models
        filename_models_list = "models_list_" + model_params + "_" + str(weight_penalty) +  "_" + ".pkl"
        with open(filename_models_list, "wb") as outfile:
            pickle.dump(models_list, outfile)
        
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
