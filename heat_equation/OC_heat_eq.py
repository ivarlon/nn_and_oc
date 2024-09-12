# -*- coding: utf-8 -*-
"""
controlling heat eq
J = 0.5*||y-y_d||^2 + 0.5*nu*||u||^2
subject to  y_t = D*y_xx + u

opt. conditions:
    -p_t = D*p_xx + y-y_d
    0 = grad(J) = nu*u + p
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

import glob
import pickle

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parents[0]
sys.path.append(str(root))
sys.path.append(str(root / "utils"))
sys.path.append(str(root / "heat_equation"))
from FNO import FNO
from DeepONet import DeepONet
from generate_data_heat_eq import solve_state_eq, solve_adjoint_eq
from optimization_routines import grad_descent, conjugate_gradient, linesearch

N_t = 64 # no. of time points
N_x = 32 # no. of spatial points
diffusion_coeff = 1e-1 # diffusion coefficient

T = 1.
t = np.linspace(0.,T, N_t)
delta_t = T/(N_t-1)

L = 2.
x = np.linspace(0.,L, N_x)
delta_x = L/(N_x-1)

tt, xx = np.meshgrid(t,x, indexing='ij')

nu = 1e-3 # penalty on control u

# desired state for OC is single peak
y_d = 0.5*np.sin(np.linspace(0., np.pi, N_t)[:,None].repeat(N_x,1))**10 

# boundary conditions on state
y_IC = 0.5*np.sin(np.linspace(0., 2*np.pi, N_x))**2 # initial condition on state is double peak with amplitude 2
y_BCs = (np.zeros(N_t), np.zeros(N_t)) # Dirichlet boundary conditions on state

# boundary conditions on adjoint
p_TC = np.zeros(N_x) # terminal condition on adjoint is zero
p_BCs = (np.zeros(N_t), np.zeros(N_t)) # zero Dirichlet boundary conditions


def get_solvers_and_functions(method_state,method_gradient,state_models_list=None,adjoint_models_list=None):
    # defining cost function to be minimised
    def cost(y, u, y_d):
        return 0.5*np.sum((y-y_d)**2) + 0.5*nu*np.sum(u**2)
    #================================================
    # Different cost functions and gradients thereof
    # are defined in the following, according to 
    # whatever method was selected.
    #================================================
    
    #---------------
    # state methods
    #---------------
    if method_state == "conventional":
        calculate_state = lambda u: solve_state_eq(u, y_IC, y_BCs, diffusion_coeff, t_span=(0.,T), x_span=(0.,L))
    
    elif method_state == "NN":
        # load saved neural operator models
        assert not isinstance(state_models_list, type(None)), "Please pass a list of state models."
        
        n_models = len(state_models_list)
        
        if model_name=="FNO":
            def calculate_state(u):
                # average over ensemble predictions
                y = 0.25*torch.cat([model(torch.tensor(u, dtype=torch.float32).unsqueeze(-1)) for model in state_models_list ])
                y = y.view(n_models,N_t,N_x)
                y = y.detach().numpy()
                return y
        else:
            tx = torch.cartesian_prod(torch.tensor(t, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))[None]
            def calculate_state(u):
                # average over ensemble predictions
                y = torch.cat([model(torch.tensor(u, dtype=torch.float32), tx) for model in state_models_list ])
                y = y.view(n_models,N_t,N_x)
                y = y.detach().numpy()
                return y
    else:
        assert False, "Please specify a valid state solver method: conventional or NN"
    
    def reduced_cost(u, y_d):
        y = calculate_state(u)
        return cost(y, u, y_d)
    
    
    #-------------------------
    # adjoint/gradient methods
    #-------------------------
    if method_gradient == "conventional adjoint":
        calculate_adjoint = lambda y, y_d: solve_adjoint_eq(y, 
                                                            y_d, 
                                                            p_TC, 
                                                            p_BCs, 
                                                            diffusion_coeff, 
                                                            t_span=(0.,T), 
                                                            x_span=(0.,L))
        def gradient_cost(u, y_d):
            y = calculate_state(u)
            p = calculate_adjoint(y, y_d)
            return nu*u + p.mean(axis=0)[None]
        
    elif method_gradient == "NN adjoint":
        
        n_adj_models = len(adjoint_models_list)
        
        if model_name == "FNO":
            def calculate_adjoint(y, y_d):
                y_y_d = torch.tensor(y-y_d, dtype=torch.float32).unsqueeze(-1)
                # calculate ensemble predictions
                p = torch.cat([model(y_y_d) for model in adjoint_models_list ])
                p = p.view(n_adj_models*y_y_d.shape[0],N_t,N_x)
                p = p.detach().numpy()
                return p
        else:
            tx = torch.cartesian_prod(torch.tensor(t, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))[None]
            def calculate_adjoint(y, y_d):
                y_y_d = torch.tensor(y-y_d, dtype=torch.float32)
                # calculate ensemble predictions
                p = torch.cat([model(y_y_d, tx) for model in adjoint_models_list ])
                p = p.view(n_adj_models*y_y_d.shape[0],N_t,N_x)
                p = p.detach().numpy()
                return p
        
        def gradient_cost(u, y_d):
            y = calculate_state(u)
            p = calculate_adjoint(y, y_d)
            return nu*u + p.mean(axis=0)[None]
    
    elif method_gradient == "NN tangent":
        assert method_state == "NN", "The NN tangent requires that you use an NN to calculate the state!"
        if model_name == "FNO":
            def gradient_cost(u, y_d):
                # Calculates gradient of cost as dJ = J_u + J_y dy/du
                # J_u = nu*u
                # dy/du J_y = grad(NN;u)*(y-y_d)
                
                u_np = u
                J_u = nu*u_np
                u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
                
                # Calculate vJp for dNN/du^T (y-y_d)
                calculate_state_vjp = lambda u: torch.stack([model(u) for model in state_models_list]).mean(axis=0)
                y, grad_u = torch.func.vjp(calculate_state_vjp, u)
                y_y_d = y-torch.tensor(y_d).flatten().unsqueeze(-1)
                dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
                y = y.detach().numpy()
                
                return J_u + dJdy_times_dydu
        else:
            # use DeepONet which takes both control u and domain (t,x) as input
            def gradient_cost(u, y_d):
                # Calculates gradient of cost as dJ = J_u + J_y dy/du
                # J_u = nu*u
                # dy/du J_y = grad(NN;u)*(y-y_d)
                
                u_np = u
                J_u = nu*u_np
                u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
                
                # Calculate vJp for dNN/du^T (y-y_d)
                calculate_state_vjp = lambda u: torch.stack([model(u,tx) for model in state_models_list]).mean(axis=0)
                y, grad_u = torch.func.vjp(calculate_state_vjp, u)
                y_y_d = y-torch.tensor(y_d).flatten().unsqueeze(-1)
                dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
                y = y.detach().numpy()
                
                return J_u + dJdy_times_dydu
    
    else:
        assert False, "Please specify a valid adjoint solver method: conventional adjoint, NN adjoint or NN tangent"
    
    
    return calculate_state, cost, reduced_cost, gradient_cost

if __name__=="__main__":
    model_name = "DON" # "DON" # NN model to use (if any)
    models = {} # stores the NN models in a dictionary

    # available methods: 
    #   state: conventional, NN
    #   adjoint: conventional adjoint, NN adjoint, NN tangent (doesn't calculate adjoint)
    method_state = "NN"
    method_gradient = "conventional adjoint"
    
    
    if method_state == "NN":
        filename_state_models = glob.glob('.\{}_state\models_list_3*.pkl'.format(model_name))[0]
        with open(filename_state_models, 'rb') as infile:
            state_models_list = pickle.load(infile)
    else:
        state_models_list = None
    
    if method_gradient == "NN adjoint":
        # load saved adjoint neural operator models
        filename_adjoint_models = glob.glob('.\{}_adjoint\models_list*.pkl'.format(model_name))[0]
        with open(filename_adjoint_models, 'rb') as infile:
            adjoint_models_list = pickle.load(infile)
    else:
        adjoint_models_list = None
    
    calculate_state, cost, reduced_cost, gradient_cost = get_solvers_and_functions(method_state, method_gradient, state_models_list, adjoint_models_list)
    
    #====================
    # Do optimal control
    #====================
        
    u0 = np.zeros_like(tt)[None] # initial guess is zero
    max_no_iters = 30 # max. no. of optimisation iterations
    
    
    # time optimisation routine
    import time
    time_start = time.time()
    
    from scipy.optimize import fmin_cg
    from scipy.optimize import minimize
    
    #res = fmin_cg(lambda u: reduced_cost(u, y_d), u0, fprime=lambda u: gradient_cost(u, y_d))
    
    #result = minimize(lambda u: reduced_cost(u, y_d), u0, method='CG', jac=lambda u: gradient_cost(u, y_d), tol=1e-4, options={'maxiter': 100, 'disp': True})
    #assert False
    u_opt, cost_history, grad_history  = grad_descent(lambda u: reduced_cost(u, y_d),
                                             lambda u: gradient_cost(u, y_d),
                                             u0,
                                             max_no_iters=max_no_iters)
    
    print()
    print("Optimisation took", round(time.time() - time_start, 1), "secs")
    
    y_opt = calculate_state(u_opt)[0]
    #y_opt = solve_state_eq(u_opt, y_IC, y_BCs, diffusion_coeff, (0.,T), (0.,L))[0]
    plt.contourf(tt, xx, y_opt, levels=np.linspace(y_opt.min(), y_opt.max())); plt.colorbar()
    plt.xlabel("t"); plt.ylabel("x")
    plt.show()
    plt.plot(np.arange(len(cost_history)), cost_history); plt.yscale("log")
    plt.xlabel("Iteration"); plt.ylabel("Cost")
    plt.show()
    
    """
    if method_state == "NN":
        # compare NN predicted state with numerical solution
        plt.plot(x, solve_state_eq(u_opt).ravel(), color="red")
    plt.plot(x, y_d.ravel())
    plt.plot(x, u_opt.ravel(), linestyle="--")
    
    """
    def taylor_test(J, u, h, J_derivative_h, rtol=1e-4):
        # J: reduced cost function
        # u: control, input to J
        # h: direction in which to compute directional derivative
        # J_derivative_h: actual directional derivative of J at u in direction h
        n = 12
        eps = 1e-3*np.logspace(0,-(n-1),n,base=2.)
        
        # compute residuals
        residuals = np.zeros(n)
        for i in range(n):
            Jh = J(u + eps[i]*h)
            residuals[i] = np.abs(Jh - J(u) - eps[i]*J_derivative_h)
        
        # compute convergence rates
        convergence_rates = np.zeros(n-1)
        for i in range(n-1):
            convergence_rates[i] = np.log( residuals[i+1]/residuals[i] ) / np.log( eps[i+1]/eps[i] )
        print(convergence_rates)
        print(np.isclose(convergence_rates, 2., rtol=rtol))