# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:56:21 2024

solving OC problem J = 0.5*||y-y_d||^2 + 0.5*nu*||u||^2
subject to  y' = -y + u

opt. conditions:
    -p' = -p + y-y_d
    0 = grad(J) = nu*u + p
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

plt.style.use("ggplot")

import glob
import pickle

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parents[0]
sys.path.append(str(root))
sys.path.append(str(root / "utils"))
sys.path.append(str(root / "simple_scalar_problem"))
from FNO import FNO
from DeepONet import DeepONet
from generate_data import solve_state_eq, solve_adjoint_eq
from optimization_routines import grad_descent, conjugate_gradient, linesearch

"""
def conjugate_gradient(f, grad_f, x0, eps=1e-8, max_no_iters=100):
    x = x0
    cost_history = []
    cost_history.append(f(x0))
    grad_history = []
    grad_history.append(np.linalg.norm(grad_f(x0)))
    i = 0
    p = -grad_f(x) # initial descent direction
    rOld = p # residual r = 0 - grad(f)
    rNew = p
    #print("Yass")
    while np.linalg.norm(grad_f(x))>eps:

        if i%(max(max_no_iters//200,1))==0:
            print("{}/{}".format(i, max_no_iters))
        if i<max_no_iters:
            s = linesearch(f, grad_f, p, x) # line search
            # update solution
            x = x + s*p
            # now update search direction p
            rNew = -grad_f(x)
            p = rNew + np.sum(rNew*rNew)/np.sum(rOld*rOld) * p
            rOld = rNew
            i+=1
            cost_history.append(f(x))
            grad_history.append(np.linalg.norm(grad_f(x)))
        else:
            break

    return x, cost_history, grad_history
"""
def linesearch(f,grad_f,p,x,starting_stepsize=4.):
    stepsize = starting_stepsize
    alpha = 0.5e-4 # desired decrease
    i = 0
    #if not f(x + stepsize*p) > f(x) + alpha*stepsize*grad_f(x).ravel()@p.ravel():
        #print("WTF")
        #print(1e3*(f(x + stepsize*p) - f(x)))
        #print(1e3*(alpha*stepsize*grad_f(x).ravel()@p.ravel()))
    while f(x + stepsize*p) > f(x) + alpha*stepsize*grad_f(x).ravel()@p.ravel():
        stepsize = 0.5*stepsize # half step size (backtracking)
        #print(i, 1e3*f(x + stepsize*p) )
        #print("", 1e3*(f(x) + alpha*stepsize*grad_f(x).ravel()@p.ravel()))
        i+=1
        print("HHHHH")
        if stepsize < 1e-5: # no decrease found
            print("shucks")
            return stepsize
    return stepsize


N = 128 # no. of grid points
nu = 1e-3 # penalty on control u
y_d = 1.5*np.ones(shape=(1,N,1)) # desired state
y0 = 1.
x = np.linspace(0.,1.,N)
delta_x = 1./(N-1)

def get_solvers_and_functions(method_state,method_gradient):
    # defining cost function to be minimised
    def cost(y, u):
        return 0.5*delta_x*np.sum((y-y_d)**2) + 0.5*nu*delta_x*np.sum(u**2)
    
    
    #================================================
    # Different cost functions and gradients thereof
    # are defined in the following, according to 
    # whatever method was selected.
    #================================================
    
    #---------------
    # state methods
    #---------------
    if method_state == "conventional":
        calculate_state = solve_state_eq
        rhs_state = lambda y, u: -y + u
    
    elif method_state == "NN":
        # load saved neural operator models
        filename = glob.glob('.\{}_state\models_list*.pkl'.format(model_name))[0]
        with open(filename, 'rb') as infile:
            models_list = pickle.load(infile)
        models["state"] = models_list
        
        if model_name == "FNO":
            def calculate_state(u):
                # calculate ensemble predictions
                y = torch.cat([model(torch.tensor(u, dtype=torch.float32)) for model in models["state"] ])
                y = y.detach().numpy()
                return y
        else:
            x = torch.linspace(0.,1.,N)[None,:,None]
            def calculate_state(u):
                 # calculate ensemble predictions
                y = torch.cat([model(torch.tensor(u, dtype=torch.float32), x) for model in models["state"] ])
                y = y.detach().numpy()
                return y
    else:
        assert False, "Please specify a valid state solver method: conventional or NN"
    
    def reduced_cost(u):
        y = calculate_state(u)
        return cost(y, u)
    
    
    #-------------------------
    # adjoint/gradient methods
    #-------------------------
    if method_gradient == "conventional adjoint":
        calculate_adjoint = lambda y: solve_adjoint_eq(y,y_d)
        def gradient_cost(u):
            y = calculate_state(u)
            p = calculate_adjoint(y)
            return nu*u + p.mean(axis=0)[None]
        
    elif method_gradient == "NN adjoint":
        # load saved adjoint neural operator models
        filename = glob.glob('.\{}_adjoint\models_list*.pkl'.format(model_name))[0]
        with open(filename, 'rb') as infile:
            models_list = pickle.load(infile)
        models["adjoint"] = models_list
        
        if model_name == "FNO":
            def calculate_adjoint(y):
                y_y_d = torch.tensor(y-y_d, dtype=torch.float32)
                # calculate ensemble predictions
                p = torch.stack([model(y_y_d) for model in models["adjoint"] ])
                p = p.detach().numpy()
                return p
        else:
            x = torch.linspace(0.,1.,N)[None,:,None]
            def calculate_adjoint(y):
                y_y_d = torch.tensor(y-y_d, dtype=torch.float32)#.view(len(y), y.shape)
                # calculate ensemble predictions
                p = torch.cat([model(y_y_d, x) for model in models["adjoint"] ])
                p = p.detach().numpy()
                return p
        
        def gradient_cost(u):
            y = calculate_state(u)
            p = calculate_adjoint(y)
            return nu*u + p.mean(axis=0)[None]
    
    elif method_gradient == "NN tangent":
        assert method_state == "NN", "The NN tangent requires that you use an NN to calculate the state!"
        if model_name == "FNO":
            def gradient_cost(u):
                # Calculates gradient of cost as dJ = J_u + J_y dy/du
                # J_u = nu*u
                # dy/du J_y = grad(NN;u)*(y-y_d)
                
                u_np = u
                J_u = nu*u_np
                u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
                
                # Calculate vJp for dNN/du^T (y-y_d)
                calculate_state_vjp = lambda u: torch.stack([model(u) for model in models["state"]]).mean(axis=0)
                y, grad_u = torch.func.vjp(calculate_state_vjp, u)
                y_y_d = y-torch.tensor(y_d)
                dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
                y = y.detach().numpy()
                
                return J_u + dJdy_times_dydu
        else:
            # Use DeepONet which takes u and x as inputs
            def gradient_cost(u, y_d):
                # Calculates gradient of cost as dJ = J_u + J_y dy/du
                # J_u = nu*u
                # dy/du J_y = grad(NN;u)*(y-y_d)
                
                u_np = u
                J_u = nu*u_np
                u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
                
                # Calculate vJp for dNN/du^T (y-y_d)
                calculate_state_vjp = lambda u: torch.stack([model(u, x) for model in models["state"]]).mean(axis=0)
                y, grad_u = torch.func.vjp(calculate_state_vjp, u)
                y_y_d = y-torch.tensor(y_d)
                dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
                y = y.detach().numpy()
                
                return J_u + dJdy_times_dydu
    else:
        assert False, "Please specify a valid adjoint solver method: conventional adjoint, NN adjoint or NN tangent"
    
    return calculate_state, calculate_adjoint, cost, reduced_cost, gradient_cost

model_name = "DON" # NN model to use (if any) = FNO or DON
models = {} # stores the NN models in a dictionary

# available methods: 
#   state: conventional, NN
#   adjoint: conventional adjoint, NN adjoint, NN tangent (doesn't calculate adjoint)
method_state = "conventional"
method_gradient = "conventional adjoint"

calculate_state, calculate_adjoint, cost, reduced_cost, gradient_cost = get_solvers_and_functions(method_state, method_gradient)

#====================
# Do optimal control
#====================
a = 10.
sigmoid = lambda x: 1./(1. + np.exp(-a*x))
u0 =  2*(y_d[0,0] - y0)*sigmoid(x)*(a - a*sigmoid(x) + 1.) + 2*y0 - y_d[0,0]# initial guess
u0 = u0.reshape(1,N,1)
u0 = np.zeros(shape=(1,N,1))
max_no_iters = 20 # max. no. of optimisation iterations

# time optimisation routine
import time
t0 = time.time()

from scipy.optimize import fmin_cg

#res = fmin_cg(reduced_cost, u0, fprime=gradient_cost)

u_history, cost_history, grad_history  = conjugate_gradient(reduced_cost,
                                         gradient_cost,
                                         u0,
                                         max_no_iters=max_no_iters,
                                         return_full_history=True,
                                         print_every=2)
u_opt = u_history[-1]
print()
print("Optimisation took", round(time.time() - t0, 1), "secs")

y_opt = calculate_state(u_opt).mean(axis=0)
p_opt = calculate_adjoint(y_opt).mean(axis=0)
x = np.linspace(0.,1.,N)

# plot state
if method_state == "NN":
    # compare NN predicted state with numerical solution
    plt.figure()
    plt.plot(x, y_opt.ravel(), label="$\\tilde{y}(u^*)$")
    plt.plot(x, solve_state_eq(u_opt).ravel(), color="red", label="$y(u^*)$")
    plt.title(model_name + ", " + method_state + " state, " + method_gradient)
else:
    plt.plot(x, y_opt.ravel(), label="opt. state")
    plt.title(method_state + " state, " + method_gradient)
plt.plot(x, y_d.ravel(), linestyle="--", label="$y_d$")
plt.xlabel("$x$"); plt.ylabel("$y$"); plt.ylim([0.95, 1.65])
plt.legend()
plt.show()

# plot optimal control
plt.figure()
plt.plot(x, u_opt.ravel(), label="$u^*$")
plt.title("Optimal control")
plt.xlabel("$x$"); plt.ylabel("$u$")
plt.legend()
plt.show()

# plot adjoint
if method_gradient == "NN adjoint":
    # compare NN predicted state with numerical solution
    plt.figure()
    plt.plot(x, p_opt.ravel(), label="$\\tilde{p}(u^*)$")
    y_actual = solve_state_eq(u_opt)
    plt.plot(x, solve_adjoint_eq(y_actual, y_d).ravel(), color="red", label="$p(u^*)$")
    plt.title(model_name + ", " + method_state + " state, " + method_gradient)
else:
    plt.plot(x, p_opt.ravel(), label="adjoint p at opt. solution")
    plt.title(method_state + " state, " + method_gradient)
plt.xlabel("$x$"); plt.ylabel("$p$")#; plt.ylim([0.95, 1.65])
plt.legend()
plt.show()

# plot cost history
plt.figure()
plt.plot(np.arange(len(cost_history)), cost_history)
if method_state == "NN":
    cost_history_true = []
    for u_ in u_history:
        y_ = solve_state_eq(u_)
        cost_ = 0.5*delta_x*np.sum((y_- y_d)**2) + 0.5*nu*delta_x*np.sum(u_**2)
        cost_history_true.append(cost_)
    plt.plot(np.arange(len(cost_history_true)), cost_history_true, linestyle="--")
else:
    filename = glob.glob('.\{}_state\models_list*.pkl'.format(model_name))[0]
    with open(filename, 'rb') as infile:
        models_list = pickle.load(infile)
    models["state"] = models_list
    def calculate_state(u):
        # calculate ensemble predictions
        y = torch.cat([model(torch.tensor(u, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)[None,:,None]) for model in models["state"] ])
        y = y.detach().numpy()
        return y
    cost_history_NN = []
    for u_ in u_history:
        y_ = calculate_state(u_)
        cost_ = 0.5*delta_x*np.sum((y_- y_d)**2) + 0.5*nu*delta_x*np.sum(u_**2)
        cost_history_NN.append(cost_)
    plt.plot(np.arange(len(cost_history_NN)), cost_history_NN, linestyle="--")
    cost_history_true = []
    calculate_state = solve_state_eq
    for u_ in u_history:
        y_ = calculate_state(u_)
        cost_ = 0.5*delta_x*np.sum((y_- y_d)**2) + 0.5*nu*delta_x*np.sum(u_**2)
        cost_history_true.append(cost_)
    plt.plot(np.arange(len(cost_history_true)), cost_history_true, linestyle="--")
plt.title("Cost")
plt.yscale("log")
plt.xlabel("It. no."); plt.ylabel("$J(u)$")
plt.show()

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
        #print(residuals[i]/eps[i])
    print(residuals)
    # compute convergence rates
    convergence_rates = np.zeros(n-1)
    for i in range(n-1):
        convergence_rates[i] = np.log( residuals[i+1]/residuals[i] ) / np.log( eps[i+1]/eps[i] )
    print(convergence_rates)
    print(np.isclose(convergence_rates, 2., rtol=rtol))

do_taylor_test = True
if do_taylor_test:
    h = 1e-2*np.ones_like(u0)
    delta_x = 1./(N-1)
    J_derivative_h = np.sum(gradient_cost(u0)*h)*delta_x
    taylor_test(reduced_cost, u0, h, J_derivative_h)