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
import glob
from FNO import FNO
from generate_data import solveStateEq, solveAdjointEq
from optimization_routines import grad_descent, conjugate_gradient, linesearch
from scipy.integrate import solve_ivp

model_name = "FNO" # NN model to use (if any)
models = {} # stores the NN models in a dictionary

# available methods: 
#   state: conventional, NN
#   adjoint: conventional adjoint, NN adjoint, NN tangent (doesn't calculate adjoint)
method_state = "conventional"
method_gradient = "conventional adjoint"

N = 64 # no. of grid points
nu = 1e-3 # penalty on control u
y_d = 1.5*np.ones(shape=(N)) # desired state

# defining cost function to be minimised
def cost(y, u, y_d):
    return 0.5*np.sum((y-y_d)**2) + 0.5*nu*np.sum(u**2)

u0 = np.zeros(shape=(N)) # initial guess
max_no_iters = 100 # max. no. of optimisation iterations


#================================================
# Different cost functions and gradients thereof
# are defined in the following, according to 
# whatever method was selected.
#================================================

#---------------
# state methods
#---------------
if method_state == "conventional":
    calculate_state = solveStateEq
    rhs_state = lambda y, u: -y + u
    calculate_state = solve_ivp(rhs_state, (0,1), 1., t_eval=np.linspace(0,1,N))

elif method_state == "NN":
    # load saved neural operator models
    filenames = glob.glob('.\data\{}_state*.pt'.format(model_name))
    models["state"] = [torch.load(filename) for filename in filenames]
    def calculate_state(u):
        # average over ensemble predictions
        y = torch.stack([model(torch.tensor(u, dtype=torch.float32)) for model in models["state"] ]).mean(axis=0)
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
    calculate_adjoint = solveAdjointEq
    def gradient_cost(u, y_d):
        y = calculate_state(u)
        p = calculate_adjoint(y, y_d)
        return nu*u + p
    
elif method_gradient == "NN adjoint":
    # load saved adjoint neural operator models
    filenames = glob.glob('.\data\{}_adjoint*.pt'.format(model_name))
    models["adjoint"] = [torch.load(filename) for filename in filenames]
    
    def calculate_adjoint(y, y_d):
        y_y_d = torch.tensor(y-y_d, dtype=torch.float32)
        # average over ensemble predictions
        p = torch.stack([model(y_y_d) for model in models["adjoint"] ]).mean(axis=0)
        p = p.detach().numpy()
        return p
    
    def gradient_cost(u, y_d):
        y = calculate_state(u)
        p = calculate_adjoint(y, y_d)
        return nu*u + p

elif method_gradient == "NN tangent":
    assert method_state == "NN", "The NN tangent requires that you use an NN to calculate the state!"
    def gradient_cost(u, y_d):
        # Calculates gradient of cost as dJ = J_u + J_y dy/du
        # J_u = nu*u
        # dy/du J_y = grad(NN;u)*(y-y_d)
        
        u_np = u
        J_u = nu*u_np
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        
        # Calculate vJp for dNN/du^T (y-y_d)
        calculate_state = lambda u: torch.stack([model(u) for model in models["state"]]).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state, u)
        y_y_d = y-torch.tensor(y_d)
        dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
        y = y.detach().numpy()
        
        return J_u + dJdy_times_dydu

else:
    assert False, "Please specify a valid adjoint solver method: conventional adjoint, NN adjoint or NN tangent"


#====================
# Do optimal control
#====================

# time optimisation routine
import time
t0 = time.time()

from scipy.optimize import fmin_cg

res = fmin_cg(lambda u: reduced_cost(u, y_d), u0, fprime=lambda u: gradient_cost(u, y_d))

u_opt, cost_history, grad_history  = grad_descent(lambda u: reduced_cost(u, y_d),
                                         lambda u: gradient_cost(u, y_d),
                                         u0,
                                         max_no_iters=max_no_iters)

print()
print("Optimisation took", round(time.time() - t0, 1), "secs")
"""
y_opt = calculate_state(u_opt)
x = np.linspace(0.,1.,N)
plt.plot(x, y_opt.ravel())
if method_state == "NN":
    # compare NN predicted state with numerical solution
    plt.plot(x, solveStateEq(u_opt).ravel(), color="red")
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