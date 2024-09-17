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

N = 128 # no. of grid points
nu = 1e-2 # penalty on control u
y_d = 1.5*np.ones(shape=(1,N,1)) # desired state
y0 = 1.
x = np.linspace(0.,1.,N)
delta_x = 1./(N-1)

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
                self.x = torch.linspace(0.,1.,N)[None,:,None]
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
                self.x = torch.linspace(0.,1.,N)[None,:,None]
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
        return 0.5*delta_x*np.sum((y-y_d)**2) + 0.5*self.nu*delta_x*np.sum(u**2)
    
    def reduced_cost(self, u):
        y = self.calculate_state(u).mean(axis=0)[None]
        return self.cost(y, u)
    
    def calculate_state_conventional(self, u):
        # uses improved forward Euler
        return solve_state_eq(u)
    
    def calculate_state_FNO(self, u):
        # calculate ensemble predictions
        y = torch.cat([model(torch.tensor(u, dtype=torch.float32)) for model in self.models["state"] ])
        y = y.detach().numpy()
        return y
    
    def calculate_state_DON(self, u):
         # calculate ensemble predictions
        y = torch.cat([model(torch.tensor(u, dtype=torch.float32), self.x) for model in self.models["state"] ])
        y = y.detach().numpy()
        return y
    
    def calculate_adjoint_conventional(self, y):
        return solve_adjoint_eq(y,y_d)
    
    def calculate_adjoint_FNO(self, y):
        y_y_d = torch.tensor(y-y_d, dtype=torch.float32)
        # calculate ensemble predictions
        p = torch.cat([model(y_y_d) for model in self.models["adjoint"] ])
        p = p.detach().numpy()
        return p
    
    def calculate_adjoint_DON(self, y):
        y_y_d = torch.tensor(y-y_d, dtype=torch.float32)#.view(len(y), y.shape)
        # calculate ensemble predictions
        p = torch.cat([model(y_y_d, self.x) for model in self.models["adjoint"] ])
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
        calculate_state_vjp = lambda u: torch.stack([model(u) for model in self.models["state"]]).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state_vjp, u)
        y_y_d = y-torch.tensor(y_d, dtype=torch.float32)
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
        calculate_state_vjp = lambda u: torch.stack([model(u, x) for model in self.models["state"]]).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state_vjp, u)
        y_y_d = y-torch.tensor(y_d, dtype=torch.float32)
        dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
        y = y.detach().numpy()
        
        return J_u + dJdy_times_dydu
    
if __name__ == "__main__":
    model_name = "DON" # NN model to use (if any) = FNO or DON
    
    method_state = "conventional"
    method_gradient = "conventional adjoint"
    
    state_filename = glob.glob('.\\state_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]
    state_models = load_models(state_filename)[:10]
    adjoint_filename = glob.glob('.\\adjoint_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]
    adjoint_models = load_models(adjoint_filename)[:10]
    
    problem = OC_problem(method_state, method_gradient, nu, state_models, adjoint_models, model_name)
    
    #====================
    # Do optimal control
    #====================
    
    u0 = np.zeros(shape=(1,N,1))
    
    max_no_iters = 50 # max. no. of optimisation iterations
    #assert False
    # time optimisation routine
    import time
    t0 = time.time()
    
    u_history, cost_history, grad_history  = conjugate_gradient(problem.reduced_cost,
                                             problem.gradient_cost,
                                             u0,
                                             max_no_iters=max_no_iters,
                                             return_full_history=True,
                                             print_every=2)
    u_opt = u_history[-1]
    print()
    print("Optimisation took", round(time.time() - t0, 1), "secs")
    
    y_opt = problem.calculate_state(u_opt).mean(axis=0)

    x = np.linspace(0.,1.,N)
    
    # plot state
    if method_state == "NN":
        # compare NN predicted state with numerical solution
        plt.figure()
        plt.plot(x, y_opt.ravel(), label="$\\tilde{y}(u^*)$")
        plt.plot(x, problem.calculate_state_conventional(u_opt).ravel(), color="red", label="$y(u^*)$")
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
    p_opt = problem.calculate_adjoint(y_opt).mean(axis=0)
    if method_gradient == "NN adjoint":
        # compare NN predicted state with numerical solution
        plt.figure()
        plt.plot(x, p_opt.ravel(), label="$\\tilde{p}(u^*)$")
        y_actual = problem.calculate_state_conventional(u_opt)
        plt.plot(x, problem.calculate_adjoint_conventional(y_actual).ravel(), color="red", label="$p(u^*)$")
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
            y_ = problem.calculate_state_conventional(u_)
            cost_ = problem.cost(y_, u_)
            cost_history_true.append(cost_)
        plt.plot(np.arange(len(cost_history_true)), cost_history_true, linestyle="--")
    """else:
        problem.load_state_models(state_filename)
        def calculate_state(u):
            # calculate ensemble predictions
            y = torch.cat([model(torch.tensor(u, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)[None,:,None]) for model in models["state"] ])
            y = y.detach().numpy()
            return y
        cost_history_NN = []
        state_history_NN = []
        for u_ in u_history:
            y_ = calculate_state(u_).mean(axis=0)[None]
            state_history_NN.append(y_)
            cost_ = 0.5*delta_x*np.sum((y_- y_d)**2) + 0.5*nu*delta_x*np.sum(u_**2)
            cost_history_NN.append(cost_)
        plt.plot(np.arange(len(cost_history_NN)), cost_history_NN, linestyle="--")
        cost_history_true = []
        state_history_true = []
        calculate_state = solve_state_eq
        for u_ in u_history:
            y_ = calculate_state(u_)
            state_history_true.append(y_)
            cost_ = 0.5*delta_x*np.sum((y_- y_d)**2) + 0.5*nu*delta_x*np.sum(u_**2)
            cost_history_true.append(cost_)
        plt.plot(np.arange(len(cost_history_true)), cost_history_true, linestyle="--")"""
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
    
    do_taylor_test = False
    if do_taylor_test:
        h = 1e-2*np.ones_like(u0)
        delta_x = 1./(N-1)
        J_derivative_h = np.sum(problem.gradient_cost(u0)*h)*delta_x
        taylor_test(problem.reduced_cost, u0, h, J_derivative_h)