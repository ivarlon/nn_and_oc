# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:56:21 2024

solving OC problem J = 0.5*||y-y_d||^2 + 0.5*nu*||u||^2
y' = y + u

opt. conditions:
    -p' = p + y-y_d
    0 = grad(J) = nu*u + p

"""
import numpy as np
import torch
import glob
from simple_exponential_problem.FNO import FNO
from simple_exponential_problem.generate_data import solveStateEq, solveAdjointEq
from simple_exponential_problem.optimization_routines import grad_descent, conjugate_gradient, linesearch

# available methods
methods = ["traditional", "NN state", "NN state + NN gradient", "NN state + NN adjoint"]
method = methods[0]

N = 64 # no. of grid points
nu = 1e-3 # penalty on control u
y_d = np.ones(N) # desired state

# defining cost function to be minimised
def cost(y, u, y_d):
    return 0.5*np.sum((y-y_d)**2) + 0.5*nu*np.sum(u**2)

u0 = np.zeros(N) # initial guess
max_no_iters = 100 # max. no. of optimisation iterations

"""
Different cost functions and gradients thereof
are defined in the following, according to 
whatever method was selected.
"""

if method=="traditional":
    def reduced_cost(u, y_d):
        return cost(solveStateEq(u), u, y_d)
    
    def gradient_cost(u, y_d):
        return nu*u + solveAdjointEq(solveStateEq(u), y_d)
    

else:
    # load saved neural operator models
    filenames = glob.glob('.\data\FNO*.pt')
    models = [torch.load(filename) for filename in filenames]
    
    def reduced_cost(u, y_d):
        # average over ensemble predictions
        y = torch.mean(torch.stack([model(torch.tensor(u, dtype=torch.float32)[None,:,None]) for model in models]), axis=0)
        y = y.detach().numpy().ravel()
        return cost(y, u, y_d)
    
    if method == "NN state":        
        def gradient_cost(u, y_d):
            # average over ensemble predictions
            y = torch.mean(torch.stack([model(torch.tensor(u, dtype=torch.float32)[None,:,None]) for model in models]), axis=0)
            y = y.detach().numpy().ravel()
            return nu*u + solveAdjointEq(y, y_d)
        
    elif method == "NN state + NN gradient":
        
        def gradient_cost(u, y_d):
            u_np = u
            u = torch.tensor(u, dtype=torch.float32, requires_grad=True)[None,:,None]
            y = torch.mean(torch.stack([model(u) for model in models]), axis=0)
            #############################################
            # IMPLEMENT JACOBIAN VECTOR PRODUCT INSTEAD #
            #############################################
            def grad_u(u): 
                dNNdu = []
                for i in range(y.shape[1]):
                    grad_outputs = torch.zeros_like(y)
                    grad_outputs[:,i] = 1.
                    dNNdu.append( torch.autograd.grad(y, u, grad_outputs=grad_outputs, retain_graph=True)[0] )
                dNNdu = torch.stack(dNNdu)[...,0]
                return dNNdu
            dNNdu = grad_u(u).detach().numpy()[:,0,:]
            y = y.detach().numpy().ravel()
            return nu*u_np + np.einsum('ij,j->i', dNNdu, y-y_d)
        
    elif method == "NN state + NN adjoint":
        # load saved adjoint models
        adjoint_filenames = glob.glob('.\data\adjoint*.pt')
        adjoint_models = [torch.load(filename) for filename in adjoint_filenames]
        
        def gradient_cost(u, y_d):
            y = torch.mean(torch.stack([model(u) for model in models]), axis=0)
            p = torch.mean(torch.stack([model(y) for model in adjoint_models]), axis=0)
            p = p.detach().numpy().ravel()
            return nu*u + p
    
    else:
        exit(0)
        
# time optimisation routine
import time
t0 = time.time()

u_opt, cost_history, grad_history  = grad_descent(lambda u: reduced_cost(u, y_d),
                                         lambda u: gradient_cost(u, y_d),
                                         u0,
                                         max_no_iters=max_no_iters)

print()
print("Optimisation took", round(time.time() - t0, 1), "secs")
# import gradient descent methods
# import pytorch state + adjoint models
# import state and adjoint solvers