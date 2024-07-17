# -*- coding: utf-8 -*-
"""
Trains Deep Operator Networks (DeepONets)
"""

# for saving data
import sys
import os
import pickle
import time # to measure training time

# import numerical libraries
import numpy as np
import torch
torch.set_default_dtype(torch.float32) # all tensors are float32

# import custom libraries and functions
from DeepONet import DeepONet
from utils.training_routines import train_DON
from CustomDataset import *
from generate_data_heat_eq import generate_data

# seed pytorch RNG
seed = 1234
torch.manual_seed(seed)
seed = 1234
torch.manual_seed(seed)

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:0")
else:
    print("Using CPU")
    device = torch.device("cpu")

train_adjoint = False # either train NN to solve adjoint or to solve state
generate_adjoint = train_adjoint
if not train_adjoint:
    data_dir_name = 'data_state'
else:
    data_dir_name = 'data_adjoint'

problem_dir = "heat_equation"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, problem_dir, data_dir_name)

diffusion_coeff = 0.25 # coefficient multiplying curvature term y_xx

N_t = 64 # number of time points t_i
N_x = 32 # number of spatial points x_j

# time span
T = 0.1
t0 = 0.; tf = t0 + T

# domain length
L = 0.2
x0 = 0.; xf = x0 + L

# boundary conditions
y_IC = 2.*torch.sin(torch.linspace(0., 2*np.pi, N_x))**2 # initial condition on state is double peak with amplitude 2
y_BCs = (torch.zeros(N_t), torch.zeros(N_t)) # Dirichlet boundary conditions on state
pf = 0. # terminal condition on adjoint
y_d = 1.5*torch.sin(torch.linspace(0., np.pi, N_t))**10 # desired state for OC is single peak

n_models = 1

################################
# Generate train and test data #
################################

n_train = 1000 # no. of training samples
n_test = 100 # no. of test samples
n_val = 100 # no. of training samples
batch_size_fun = 100 # minibatch size during SGD
batch_size_loc = N_x*N_t # no. of minibatch domain points. Get worse performance when not using entire domain :/
n_t_coeffs = 4
n_x_coeffs = 5
u_max = 10. # maximum amplitude of control

generate_data_func = lambda n_samples: generate_data(N_t, N_x, t_span=(t0,tf), x_span=(x0,xf),
                  y_IC=y_IC,
                  y_BCs=y_BCs,
                  n_t_coeffs=n_t_coeffs,
                  n_x_coeffs=n_x_coeffs,
                  n_samples=n_samples,
                  u_max=u_max,
                  diffusion_coeff=diffusion_coeff,
                  generate_adjoint=generate_adjoint)

# generate different data for different models?
different_data = False
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data_func(n_train)
        data["tx"].requires_grad = True
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data_func(n_train)
    data["tx"].requires_grad = True
    train_data = n_models*[data]

# generate test and validation data
test_data = generate_data_func(n_test)
test_data["tx"].requires_grad = True
u_test = test_data["u"]; tx_test = test_data["tx"]; y_test = test_data["y"]

val_data = generate_data_func(n_val)
val_data["tx"].requires_grad = True
dataset_val = (val_data["u"].to(device), val_data["tx"].to(device), val_data["y"].to(device))

# data augmentation: do linear combinations of solutions (heat eq is linear)
# draw (n_augmented_samples, n_terms) coeffs from n_terms-dim hypercube [-b,b]^(n_terms)
# coeffs = torch.rand(low=-b, high=b, size=(n_augmented_samples,n_terms))
# scale to 1: coeffs = coeffs/torch.sum(coeffs, axis=1, keepdims=True). this makes it satisfy BCs
# new_samples = torch.einsum( ... )
# for n_terms in [4, 8, 16, 32]:
#       do above
# also add y += y_homogeneous to N_ 



################
# physics loss #
################
if not train_adjoint:
    def PDE_interior(u,x,y):
        # u is size (n_fun_samples, N_t*N_x)
        # x is size (n_fun_samples, N_t*N_x, 2)
        # y is size (n_fun_samples, N_t*N_x)
        dy = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        dy_t = dy[...,0]
        dy_xx = torch.autograd.grad(outputs=dy[...,1], inputs=x, grad_outputs=torch.ones_like(dy[...,1]), create_graph=True, retain_graph=True)[0]
        dy_xx = dy_xx[...,1]
        dy_t = dy_t.view(y.shape[0], N_t, N_x)
        dy_xx = dy_xx.view(y.shape[0], N_t, N_x)
        return dy_t[:,1:,1:-1] - diffusion_coeff*dy_xx[:,1:,1:-1] - u.view_as(dy_t)[:,1:,1:-1]

def physics_loss(u, x, y, weight_IC=5., weight_BC=1.):
    # y = y(x;u) is output of DeepONet, tensor of shape (n_samples, n_domain_points, dim(Y))
    # x is input tensor and has shape (n_samples, n_domain_points, dim(X))
    interior_loss = (PDE_interior(u, x, y)**2).mean()
    
    n_fun_samples = y.shape[0]
    y_reshaped = y.view(n_fun_samples, N_t, N_x) # (n_samples, N_t, N_x)
    
    y_IC_tensor = y_IC.repeat(n_fun_samples, 1)
    IC_loss = torch.nn.MSELoss()(y_reshaped[:, 0], y_IC_tensor)
    
    BC_loss = torch.nn.MSELoss()(y_reshaped[:, :, 0], y_BCs[0].repeat(n_fun_samples,1)) \
        + torch.nn.MSELoss()(y_reshaped[:, :, -1], y_BCs[1].repeat(n_fun_samples,1))
    #print("haha", interior_loss.dtype, IC_loss.dtype, BC_loss.dtype)
    return interior_loss + weight_IC*IC_loss + weight_BC*BC_loss


#######################################
# Set up and train the various models #
#######################################

model_name = "DeepONet"
input_size_branch = (N_t, N_x)
input_size_trunk = 2
n_conv_layers = 2

final_layer_size1 = 10
final_layer_size2 = 40
branch_hidden_sizes = [2*input_size_branch, 4*input_size_branch]
trunk_hidden_sizes = [5, 10, 50]

architectures = [ [[branch_hidden_sizes[0],final_layer_size1], [trunk_hidden, final_layer_size1]] for trunk_hidden in trunk_hidden_sizes[:-1]] \
    + [ [ [branch_hidden,final_layer_size2], [trunk_hidden_sizes[-1],final_layer_size2] ] for branch_hidden in branch_hidden_sizes[1:]] \
        + [ [[500, 500, 50], [50, 50]] ]
architectures = [([100,40], [100,40])]
activation_branch = torch.nn.ReLU()
activation_trunk = torch.nn.Sigmoid()

# weights for physics and data loss: loss = w_ph*loss_ph + w_d*loss_d
weight_physics = 0.5
weight_data = 1. - weight_physics


def loss_fn(preds, targets, u, x):
    loss_physics = physics_loss(u,x,preds)
    #print("predsshape", preds.view_as(targets).shape, "targetsshape", targets.shape)
    loss_data = torch.nn.MSELoss()(preds.view_as(targets), targets)
    return weight_physics*loss_physics + weight_data*loss_data

weight_penalties = [0.] # L2 penalty for NN weights


iterations = 1000 # no. of training epochs



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
            tx_train = data["tx"]
            dataset = DeepONetDataset(u_train, tx_train, y_train, device=device)
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
            #with torch.autograd.profiler.profile() as prof:
    

    
            loss_history = train_DON(model, 
                                dataset,
                                dataset_val,
                                iterations, 
                                loss_fn,
                                batch_size_fun=batch_size_fun,
                                batch_size_loc=batch_size_loc,
                                lr=3e-3,
                                weight_penalty=weight_penalty)
            #print(prof.key_averages().table(sort_by="cpu_time_total"))
            time_end = time.time()
            training_time = round(time_end-time_start,1)
            training_times.append(training_time)
            model.to('cpu')
            models_list.append(model)
            loss_histories.append(loss_history.to('cpu'))
            preds = model(u_test, tx_test)
            test_loss.append(loss_fn(preds, y_test, u_test, tx_test).item())
            print()
        print("Test losses", test_loss)
        print("R2", [1. - loss/(y_test**2).mean() for loss in test_loss])
        # save training_loss
        filename_loss_history = "loss_history_" + model_params + "_" + str(weight_penalty) + "_" + model_name + ".pkl"
        with open(os.path.join(data_dir, filename_loss_history), "wb") as outfile:
            pickle.dump(loss_histories, outfile)
        # save test loss
        filename_test_loss = "test_loss_" + model_params + "_" + str(weight_penalty) +  "_" + ".pkl"
        with open(os.path.join(data_dir, filename_test_loss), "wb") as outfile:
            pickle.dump(test_loss, outfile)
        # save models
        filename_models_list = "models_list_" + model_params + "_" + str(weight_penalty) +  "_" + ".pkl"
        with open(os.path.join(data_dir, filename_models_list), "wb") as outfile:
            pickle.dump(models_list, outfile)
        # save training time
        filename_training_times = "training_times_" + model_params + "_" + str(weight_penalty) +  "_" + ".pkl"
        with open(os.path.join(data_dir, filename_training_times), "wb") as outfile:
            pickle.dump(training_times, outfile)

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
