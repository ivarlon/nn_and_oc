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

nu = 5e-3 # penalty on control u

# desired state for OC is single peak
y_d = 0.5*np.sin(np.linspace(0., np.pi, N_x)[None].repeat(N_t,0))**10 

# boundary conditions on state
y_IC = 0.5*np.sin(np.linspace(0., 2*np.pi, N_x))**2 # initial condition on state is double peak with amplitude 2
y_BCs = (np.zeros(N_t), np.zeros(N_t)) # Dirichlet boundary conditions on state

# boundary conditions on adjoint
p_TC = np.zeros(N_x) # terminal condition on adjoint is zero
p_BCs = (np.zeros(N_t), np.zeros(N_t)) # zero Dirichlet boundary conditions

def load_models(filename):
    with open(filename, 'rb') as infile:
        models_list = pickle.load(infile)
    return models_list

class OC_problem:
    # This class defines functions to be used in the OC problem
    def __init__(self, method_state, method_gradient, nu, state_models=None, adjoint_models=None, model_name=None):
        """
        method_state (str) : method by which to calculate state y
                either 'conventional' or 'NN'
        method_gradient (str) : method by which the gradient dJ/du is computed
                either 'conventional adjoint', 'NN adjoint', 'NN tangent' (the latter doesn't calculate adjoint state)
        """
        # penalty on control
        self.nu = nu
        
        # create dictionary to hold models
        self.models = dict(state = state_models,
                           adjoint = adjoint_models)
        
        # state methods
        if method_state == "conventional":
            self.calculate_state = self.calculate_state_conventional
        
        elif method_state == "NN":
            if model_name == "FNO":
                self.calculate_state = self.calculate_state_FNO
            
            elif model_name == "DON":
                self.tx = torch.cartesian_prod(torch.tensor(t, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))[None]
                self.calculate_state = self.calculate_state_DON
            
            else:
                assert False, "Please enter a valid model name: 'DON' or 'FNO'."
      
        else:
            assert False, "Please specify a valid state solver method: conventional or NN"
        
        # gradient methods
        if method_gradient == "conventional adjoint":
            self.calculate_adjoint = self.calculate_adjoint_conventional
            self.gradient_cost = self.gradient_cost_adjoint_method
        
        elif method_gradient == "NN adjoint":
            self.gradient_cost = self.gradient_cost_adjoint_method
            if model_name == "FNO":
                self.calculate_adjoint = self.calculate_adjoint_FNO
            elif model_name == "DON":
                self.tx = torch.cartesian_prod(torch.tensor(t, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))[None]
                self.calculate_adjoint = self.calculate_adjoint_DON
            else:
                assert False, "Please enter a valid model name: 'DON' or 'FNO'."
        
        elif method_gradient == "NN tangent":
            assert method_state == "NN", "The NN tangent requires that you use an NN to calculate the state!"
            if model_name == "FNO":
                self.gradient_cost = self.FNO_tangent
            else:
                self.gradient_cost = self.DON_tangent
        else:
            assert False, "Please specify a valid adjoint solver method: conventional adjoint, NN adjoint or NN tangent"

    def cost(self, y, u):
        # cost function to be minimised
        return 0.5*delta_x*delta_t*np.sum((y-y_d)**2) + 0.5*self.nu*delta_x*delta_t*np.sum(u**2)
    
    def reduced_cost(self, u):
        y = self.calculate_state(u).mean(axis=0)[None]
        return self.cost(y, u)
    
    def calculate_state_conventional(self, u):
        # uses Crank-Nicolson
        return solve_state_eq(u, y_IC, y_BCs, diffusion_coeff, t_span=(0.,T), x_span=(0.,L))
    
    def calculate_state_FNO(self, u):
        # calculate ensemble predictions
        y = torch.cat([model(torch.tensor(u, dtype=torch.float32).unsqueeze(-1)) for model in self.models["state"] ])
        y = y.view(len(y),N_t,N_x)
        y = y.detach().numpy()
        return y
    
    def calculate_state_DON(self, u):
        # calculate ensemble predictions
        y = torch.cat([model(torch.tensor(u, dtype=torch.float32), self.tx) for model in self.models["state"] ])
        y = y.view(len(y),N_t,N_x)
        y = y.detach().numpy()
        return y
    
    def calculate_adjoint_conventional(self, y):
        return solve_adjoint_eq(y, y_d, p_TC, p_BCs, diffusion_coeff, t_span=(0.,T), x_span=(0.,L))
    
    def calculate_adjoint_FNO(self, y):
        y_y_d = torch.tensor(y-y_d, dtype=torch.float32).unsqueeze(-1)
        # calculate ensemble predictions
        p = torch.cat([model(y_y_d) for model in self.models["adjoint"] ])
        p = p.view(len(p),N_t,N_x)
        p = p.detach().numpy()
        return p
    
    def calculate_adjoint_DON(self, y):
        y_y_d = torch.tensor(y-y_d, dtype=torch.float32)
        # calculate ensemble predictions
        p = torch.cat([model(y_y_d, self.tx) for model in self.models["adjoint"] ])
        p = p.view(len(p),N_t,N_x)
        p = p.detach().numpy()
        return p
    
    def gradient_cost_adjoint_method(self, u):
        y = self.calculate_state(u)
        p = self.calculate_adjoint(y)
        return self.nu*u + p.mean(axis=0)[None]
    
    def FNO_tangent(self, u):
        # Calculates gradient of cost as dJ = J_u + J_y dy/du
        # J_u = nu*u
        # dy/du J_y = grad(NN;u)*(y-y_d)
        u_np = u
        J_u = self.nu*u_np
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        
        # Calculate vJp for dNN/du^T (y-y_d)
        calculate_state_vjp = lambda u: torch.stack([model(u) for model in self.models["state"] ]).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state_vjp, u)
        y_y_d = y-torch.tensor(y_d, dtype=torch.float32).flatten().unsqueeze(-1)
        dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
        y = y.detach().numpy()
        
        return J_u + dJdy_times_dydu
    
    def DON_tangent(self, u):
        # Calculates gradient of cost as dJ = J_u + J_y dy/du
        # J_u = nu*u
        # dy/du J_y = grad(NN;u)*(y-y_d)
        
        u_np = u
        J_u = self.nu*u_np
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        
        # Calculate vJp for dNN/du^T (y-y_d)
        calculate_state_vjp = lambda u: torch.stack([model(u,self.tx) for model in self.models["state"] ]).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state_vjp, u)
        y_y_d = y-torch.tensor(y_d).flatten().unsqueeze(-1)
        dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
        y = y.detach().numpy()
        
        return J_u + dJdy_times_dydu

if __name__=="__main__":
    model_name = "DON" # "DON" # NN model to use (if any)
    
    method_state = "NN"
    method_gradient = "NN tangent"
    
    state_filename = glob.glob('.\{}_state\models_list*.pkl'.format(model_name))[0]
    state_models = load_models(state_filename)
    #adjoint_filename = glob.glob('.\\adjoint_experiments_{}\\models_list_2_32_0.001.pkl'.format(model_name))[0]
    #adjoint_models = load_models(adjoint_filename)
    adjoint_models=None
    
    problem = OC_problem(method_state, method_gradient, nu, state_models, adjoint_models, model_name)
    
    
    #====================
    # Do optimal control
    #====================
        
    u0 = np.zeros_like(tt)[None] # initial guess is zero
    max_no_iters = 10 # max. no. of optimisation iterations
    
    
    # time optimisation routine
    import time
    time_start = time.time()
    
    u_history, cost_history, grad_history  = grad_descent(problem.reduced_cost,
                                             problem.gradient_cost,
                                             u0,
                                             max_no_iters=max_no_iters,
                                             return_full_history=True,
                                             print_every=2)
    
    print()
    print("Optimisation took", round(time.time() - time_start, 1), "secs")
    
    u_opt = u_history[-1]
    
    # plot optimal state
    y_opt = problem.calculate_state(u_opt)#.mean(axis=0)
    fig = plt.figure(figsize=(8 + 8*(method_state=="NN"),8))
    ax0 = fig.add_subplot(111) if method_state=="conventional" else fig.add_subplot(121)
    contour0 = ax0.contourf(tt, xx, y_opt.mean(axis=0), levels=np.linspace(y_opt.min(), y_opt.max()))
    fig.colorbar(contour0, ax=ax0, label="$y$")
    ax0.set_xlabel("$t$"); ax0.set_ylabel("$x$")
    ax0.set_title("Optimal state, " + model_name*(method_state=="NN") + "conventional solver"*(method_state!="NN"))
    
    
    if method_state=="NN":
        ax1 = fig.add_subplot(122)
        y_opt_actual = problem.calculate_state_conventional(u_opt).mean(axis=0)
        #rel_error = np.abs((y_opt - y_opt_actual)).mean(axis=0)#/(y_opt_actual + 1e-7))
        #contour1 = ax1.contourf(tt, xx, rel_error**0.5, levels=np.linspace((rel_error[rel_error<1e2]**0.5).min(), (rel_error[rel_error<1e2]**0.5).max()))
        contour1 = ax1.contourf(tt, xx, y_opt_actual, levels=np.linspace(y_opt_actual.min(), y_opt_actual.max()))
        fig.colorbar(contour1, ax=ax1, label="$|\\tilde{y} - y|$")
        ax1.set_xlabel("$t$"); ax1.set_ylabel("$x$")
        ax1.set_title("Difference NO prediction and numerical solution")
    fig.tight_layout() 
    plt.show()
    
    """
    if method_state=="NN":
        fig, axs = plt.subplots(ncols=min(3,len(y_opt)), figsize=(18,6))
        from matplotlib.ticker import FormatStrFormatter
        contours = []
        for i in range(min(3,len(y_opt))):
            contours.append(axs[i].contourf(tt, xx, y_opt[i], levels=np.linspace(y_opt[i].min(), y_opt[i].max())))
            cbar = fig.colorbar(contours[i], ax=axs[i])
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.tight_layout()
        plt.show()
    """
    # plot optimal control
    fig, ax = plt.subplots(figsize=(8,8))
    contour = ax.contourf(tt, xx, u_opt[0], levels=np.linspace(u_opt.min(), u_opt.max()))
    fig.colorbar(contour, ax=ax, label="$u$")
    ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
    ax.set_title("Optimal control, " + model_name*(method_state=="NN") + "conventional solver"*(method_state!="NN"))
    plt.show()
    
    # plot cost history
    fig = plt.figure(figsize=(8,6))
    cost_plot = plt.plot(np.arange(len(cost_history)), cost_history)[0]
    if method_state == "NN":
        cost_plot.set_label("$J(\\tilde{y}, u)$")
        cost_history_true = []
        for u_ in u_history:
            y_ = problem.calculate_state_conventional(u_)
            cost_ = problem.cost(y_, u_)
            cost_history_true.append(cost_)
        plt.plot(np.arange(len(cost_history_true)), cost_history_true, linestyle="--", label="$J(y,u)$")
    else:
        cost_plot.set_label("$J(y, u)$")
    plt.xlabel("Iteration"); plt.ylabel("Cost"); plt.yscale("log")
    plt.legend()
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