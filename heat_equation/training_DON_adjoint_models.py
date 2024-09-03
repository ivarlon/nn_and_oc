# -*- coding: utf-8 -*-
"""
Trains Deep Operator Networks (DeepONets) to solve adjoint eq. for control of heat eq.
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
from generate_data_heat_eq import generate_data, augment_data

# seed pytorch RNG
seed = 4321
torch.manual_seed(seed)

try:
    cuda = int(sys.argv[-1])
except:
    cuda = 0 # 1,2,3

if torch.cuda.is_available():
    print("Using CUDA", cuda)
    device = torch.device("cuda:{}".format(cuda))
else:
    print("Using CPU\n")
    device = torch.device("cpu")

# create data directory to store models and results
data_dir_name = 'adjoint_experiments_DON_hehe'
problem_dir_name = "heat_equation"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, problem_dir_name, data_dir_name)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


diffusion_coeff = 1e-1 # coefficient multiplying curvature term y_xx

N_t = 64 # number of time points t_i
N_x = 32 # number of spatial points x_j

# time span
T = 1.
t0 = 0.; tf = t0 + T

# domain length
L = 2.
x0 = 0.; xf = x0 + L

# boundary conditions
p_TC = torch.zeros(N_x) # terminal condition on adjoint is zero
p_BCs = (torch.zeros(N_t), torch.zeros(N_t)) # zero Dirichlet boundary conditions

y_d = 0.5*torch.sin(torch.linspace(0., np.pi, N_t)[:,None].repeat(1,N_x))**10 # desired state for OC is single peak

n_models = 3

################################
# Generate train and test data #
################################

n_train = 5000 # no. of training samples
n_test = 500 # no. of test samples
n_val = 400 # no. of training samples
batch_size_fun = 50 # minibatch size during SGD
batch_size_loc = N_x*N_t # no. of minibatch domain points. Get worse performance when not using entire domain :/
n_t_coeffs = 4
n_x_coeffs = 5
y_yd_max = 10. # maximum amplitude of y-y_d used for training

generate_data_func = lambda n_samples: generate_data(N_t, N_x, t_span=(t0,tf), x_span=(x0,xf),
                  IC=p_TC,
                  BCs=p_BCs,
                  n_t_coeffs=n_t_coeffs,
                  n_x_coeffs=n_x_coeffs,
                  n_samples=n_samples,
                  u_max=y_yd_max,
                  diffusion_coeff=diffusion_coeff,
                  y_d=y_d.numpy(),
                  generate_adjoint=True)

# generate different data for different models?
different_data = True
if different_data:
    train_data = []
    for m in range(n_models):
        data = generate_data_func(n_train-2000)
        augment_data(data, n_augmented_samples=2000, n_combinations=5, max_coeff=2, adjoint=True)
        data["tx"].requires_grad = True
        train_data.append(data)
else:
    # use the same training data for all models
    data = generate_data_func(n_train-2000)
    augment_data(data, n_augmented_samples=2000, n_combinations=5, max_coeff=2, adjoint=True)
    data["tx"].requires_grad = True
    train_data = n_models*[data]

# generate test and validation data
test_data = generate_data_func(n_test-200)
augment_data(test_data, n_augmented_samples=200, n_combinations=5, max_coeff=2, adjoint=True)
test_data["tx"].requires_grad = True
y_y_d_test = test_data["y-y_d"]; tx_test = test_data["tx"]; p_test = test_data["p"]

val_data = generate_data_func(n_val-200)
augment_data(val_data, n_augmented_samples=200, n_combinations=5, max_coeff=2, adjoint=True)
val_data["tx"].requires_grad = True
dataset_val = (val_data["y-y_d"].to(device), val_data["tx"].to(device), val_data["p"].to(device))


################
# physics loss #
################

def PDE_interior(y,x,p):
    # y is size (n_fun_samples, N_t*N_x)
    # x is size (n_fun_samples, N_t*N_x, 2)
    # p is size (n_fun_samples, N_t*N_x)
    dp = torch.autograd.grad(outputs=p, inputs=x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    dp_t = dp[...,0]
    dp_xx = torch.autograd.grad(outputs=dp[...,1], inputs=x, grad_outputs=torch.ones_like(dp[...,1]), create_graph=True, retain_graph=True)[0]
    dp_xx = dp_xx[...,1]
    dp_t = dp_t.view(p.shape[0], N_t, N_x)
    dp_xx = dp_xx.view(p.shape[0], N_t, N_x)
    return dp_t[:,:-1,1:-1] + diffusion_coeff*dp_xx[:,:-1,1:-1] + y.view_as(dp_t)[:,:-1,1:-1]

def physics_loss(y, x, p, p_TC, p_BCs, weight_TC=5., weight_BC=1.):
    # p = p(x;y) is output of DeepONet, tensor of shape (n_samples, n_domain_points, dim(p))
    # x is input tensor and has shape (n_samples, n_domain_points, dim(X))
    interior_loss = (PDE_interior(y, x, p)**2).mean()
    
    n_fun_samples = p.shape[0]
    p_reshaped = p.view(n_fun_samples, N_t, N_x) # (n_samples, N_t, N_x)
    
    p_TC_tensor = p_TC.repeat(n_fun_samples, 1)
    TC_loss = torch.nn.MSELoss()(p_reshaped[:, -1], p_TC_tensor)
    
    BC_loss = torch.nn.MSELoss()(p_reshaped[:, :, 0], p_BCs[0].repeat(n_fun_samples,1)) \
        + torch.nn.MSELoss()(p_reshaped[:, :, -1], p_BCs[1].repeat(n_fun_samples,1))
    
    return interior_loss + weight_TC*TC_loss + weight_BC*BC_loss


#######################################
# Set up and train the various models #
#######################################

input_size_branch = (N_t, N_x)
input_size_trunk = 2


architectures = [([100,40], [100,40]),
                 ([100,100,40], [100,40]),
                 ([100,40], [100,100,40]),
                 ([200,100], [200,100]),
                 ([200,200,100], [200,100]  ) ][cuda:cuda+1]
n_conv_layers_list = [0,3][1:]

activation_branch = torch.nn.ReLU()
activation_trunk = torch.nn.Sigmoid()

# weights for physics and data loss: loss = w_ph*loss_ph + w_d*loss_d
weight_physics = 0.5
weight_data = 1. - weight_physics

# define loss functions (one for CPU evaluation, the other for device (cuda or CPU))
p_TC = p_TC.to(device)
p_BCs = (p_BCs[0].to(device), p_BCs[1].to(device))


loss_fn_physics_CPU = lambda preds, targets, y, x: weight_physics * physics_loss(y,x,preds,p_TC.to('cpu'),(p_BCs[0].to('cpu'), p_BCs[1].to('cpu')))
loss_fn_physics_device = lambda preds, targets, y, x: weight_physics * physics_loss(y,x,preds,p_TC,p_BCs)

loss_fn_data = lambda preds, targets, y, x: weight_data * torch.nn.MSELoss()(preds.view_as(targets), targets)


weight_penalty = 0. # L2 penalty for NN weights
learning_rates = [1e-3]

iterations = 5000 # no. of training epochs


"""
Train the various models
"""
retrain_if_low_r2 = False # retrain model one additional time if R2 on test set is below desired score. The model is discarded and a new one initialised if the retrain still yields R2<0.95.
max_n_retrains = 20 # max. no. of retrains (to avoid potential infinite retrain loop)
desired_r2 = 0.99

for n_conv_layers in n_conv_layers_list:
    print("Using", n_conv_layers, "conv layers")
    for architecture in architectures:
        branch_architecture, trunk_architecture = architecture
        model_params = str(branch_architecture) + "_" + str(trunk_architecture)
        print("Training with branch architecture", branch_architecture, "\nTrunk architecture", trunk_architecture, "\n")
        
        for lr in learning_rates:
            print("Using learning rate", lr, "\n")
            
            models_list = [] # list to store models
            
            # create metrics dict
            metrics = dict(test_loss = [], 
                           R2 = [],
                           training_times=[])
            
            # create dict to store the training loss histories
            loss_histories = dict(total = [], 
                              data = [],
                              physics = [],
                              validation = [])
            
            m = 0
            n_retrains = 0
            while m < n_models:
                m += 1
                print("Training model", str(m) + "/" + str(n_models))
                data = train_data[m-1]
                y_y_d_train = data["y-y_d"]
                p_train = data["p"]
                tx_train = data["tx"]
                dataset = DeepONetDataset(y_y_d_train, tx_train, p_train, device=device)
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
                
                loss_hist, loss_data_hist, loss_physics_hist, loss_hist_val = train_DON(model, 
                                                                                dataset,
                                                                                dataset_val,
                                                                                iterations, 
                                                                                loss_fn_data,
                                                                                loss_fn_physics_device,
                                                                                batch_size_fun=batch_size_fun,
                                                                                batch_size_loc=batch_size_loc,
                                                                                lr=lr,
                                                                                weight_penalty=weight_penalty,
                                                                                device=device,
                                                                                print_every=20)
                
                time_end = time.time()
                training_time = time_end - time_start
                
                model.to('cpu')
                preds = model(y_y_d_test, tx_test)
                
                r2 = 1. - torch.mean(((preds.flatten(start_dim=1)-p_test.flatten(start_dim=1))**2).mean(axis=1)/p_test.flatten(start_dim=1).var(axis=1))
                if retrain_if_low_r2:
                    if r2 < desired_r2:
                        print("R2 = {:.2f} < {:.2f}, retraining for {:g} epochs.".format(r2, desired_r2, iterations))
                        n_retrains += 1
                        model.to(device)
                        time_start = time.time()
                        loss_hist_new, loss_data_hist_new, loss_physics_hist_new, loss_hist_val_new = train_DON(model, 
                                                                                                            dataset,
                                                                                                            dataset_val,
                                                                                                            iterations, 
                                                                                                            loss_fn_data,
                                                                                                            loss_fn_physics_device,
                                                                                                            batch_size_fun=batch_size_fun,
                                                                                                            batch_size_loc=batch_size_loc,
                                                                                                            lr=lr,
                                                                                                            weight_penalty=weight_penalty,
                                                                                                            device=device,
                                                                                                            print_every=20)
                        time_end = time.time()
                        training_time = training_time + time_end - time_start
                        
                        loss_hist = torch.cat((loss_hist, loss_hist_new))
                        loss_data_hist = torch.cat((loss_data_hist, loss_data_hist_new))
                        loss_physics_hist = torch.cat((loss_physics_hist, loss_physics_hist_new))
                        loss_hist_val = torch.cat((loss_hist_val, loss_hist_val_new))
                        
                        model.to("cpu")
                        preds = model(y_y_d_test, tx_test)
                        
                        r2 = 1. - torch.mean(((preds.flatten(start_dim=1)-p_test.flatten(start_dim=1))**2).mean(axis=1)/p_test.flatten(start_dim=1).var(axis=1))
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
                
                # calculate test losses
                test_loss_data = torch.nn.MSELoss()(preds.view_as(p_test), p_test).item()
                test_loss_physics = physics_loss(y_y_d_test, tx_test, preds, p_TC.to('cpu'),(p_BCs[0].to('cpu'), p_BCs[1].to('cpu')))
                test_losses = (test_loss_data, test_loss_physics)
                
                metrics["test_loss"].append(test_losses)
                metrics["R2"].append( r2 )
                metrics["training_times"].append(training_time)
                
                models_list.append(model)
                
                loss_histories["total"].append(loss_hist.to('cpu'))
                loss_histories["data"].append(loss_data_hist.to('cpu'))
                loss_histories["physics"].append(loss_physics_hist.to('cpu'))
                loss_histories["validation"].append(loss_hist_val.to('cpu'))
                
                
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