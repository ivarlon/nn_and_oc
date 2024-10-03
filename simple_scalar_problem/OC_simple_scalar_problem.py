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

from torch.func import stack_module_state, functional_call, vmap # to vectorise ensemble
import copy

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
nu = 5e-3 # penalty on control u
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
        
        # initalise ensemble of models
        self.initialise_ensemble(state_models, adjoint_models)
        
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
    
    def initialise_ensemble(self, state_models, adjoint_models):
        
        if state_models is not None:
            self.state_params, self.state_buffers = stack_module_state(state_models)
            self.base_model_state = copy.deepcopy(state_models[0]).to("meta") # make a "meta" version of models
        
        if adjoint_models is not None:
            self.adjoint_params, self.adjoint_buffers = stack_module_state(adjoint_models)
            self.base_model_adjoint = copy.deepcopy(adjoint_models[0]).to("meta") # make a "meta" version of models
    
    def calculate_state_conventional(self, u):
        # uses improved forward Euler
        return solve_state_eq(u)
    
    def calculate_state_FNO(self, u):
        # calculate ensemble predictions
        def state_ensemble(params, buffers, u):
            return functional_call( self.base_model_state, (params, buffers), (u,))
        
        y = vmap(state_ensemble, in_dims=(0, 0, None))(self.state_params, self.state_buffers, torch.tensor(u, dtype=torch.float32).unsqueeze(-1))
        y = y.view(len(y),N,1)
        y = y.detach().numpy()
        return y
    
    def calculate_state_DON(self, u):
         # calculate ensemble predictions
        def state_ensemble(params, buffers, u):
            return functional_call( self.base_model_state, (params, buffers), (u, self.x))
        
        y = vmap(state_ensemble, in_dims=(0, 0, None))(self.state_params, self.state_buffers, torch.tensor(u, dtype=torch.float32))
        y = y.view(len(y),N,1)
        y = y.detach().numpy()
        return y
    
    def calculate_adjoint_conventional(self, y):
        return solve_adjoint_eq(y,y_d)
    
    def calculate_adjoint_FNO(self, y):
        y_y_d = torch.tensor(y-y_d, dtype=torch.float32)
        # calculate ensemble predictions
        def adjoint_ensemble(params, buffers, y):
            return functional_call( self.base_model_adjoint, (params, buffers), (y,))
        
        p = vmap(adjoint_ensemble, in_dims=(0, 0, None))(self.adjoint_params, self.adjoint_buffers, y_y_d)
        p = p.view(len(p)*len(y_y_d), N, 1)
        p = p.detach().numpy()
        return p
    
    def calculate_adjoint_DON(self, y):
        y_y_d = torch.tensor(y-y_d, dtype=torch.float32)#.view(len(y), y.shape)
        # calculate ensemble predictions
        def adjoint_ensemble(params, buffers, y):
            return functional_call( self.base_model_adjoint, (params, buffers), (y, self.x))
        
        p = vmap(adjoint_ensemble, in_dims=(0, 0, None))(self.adjoint_params, self.adjoint_buffers, y_y_d)
        p = p.view(len(p)*len(y_y_d), N, 1)
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
        def state_ensemble(params, buffers, u):
            return functional_call( self.base_model_state, (params, buffers), (u,))
        
        # Calculate vJp for dNN/du^T (y-y_d)
        calculate_state_vjp = lambda u: vmap(state_ensemble, in_dims=(0, 0, None))(self.state_params, self.state_buffers, u).mean(axis=0)
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
        def state_ensemble(params, buffers, u):
            return functional_call( self.base_model_state, (params, buffers), (u, self.x))
        
        # Calculate vJp for dNN/du^T (y-y_d)
        calculate_state_vjp = lambda u: vmap(state_ensemble, in_dims=(0, 0, None))(self.state_params, self.state_buffers, u).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state_vjp, u)
        y_y_d = y-torch.tensor(y_d, dtype=torch.float32)
        dJdy_times_dydu = grad_u(y_y_d)[0].detach().numpy()
        y = y.detach().numpy()
        
        return J_u + dJdy_times_dydu
    
if __name__ == "__main__":
    savefigs = True
    figsize = (3.5,3)
    model_name = "FNO" # NN model to use (if any) = FNO or DON
    
    method_state = "NN"
    method_gradient = "NN adjoint"
    
    state_filename = glob.glob('.\\state_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]
    state_models = load_models(state_filename)[:]
    adjoint_filename = glob.glob('.\\adjoint_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]
    adjoint_models = load_models(adjoint_filename)[:]
    
    problem = OC_problem(method_state, method_gradient, nu, state_models, adjoint_models, model_name)
    
    #====================
    # Do optimal control
    #====================
    
    u0 = np.zeros(shape=(1,N,1))
    
    max_no_iters = 20 # max. no. of optimisation iterations
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
    
    y_opt = problem.calculate_state(u_opt)

    x = np.linspace(0.,1.,N)
    
    # plot state
    if method_state == "NN":
        # compare NN predicted state with numerical solution
        plt.figure(figsize=figsize)
        plt.plot(x, y_opt.mean(axis=0).ravel(), label="$\\tilde{y}(u^*)$")
        y_opt_actual = problem.calculate_state_conventional(u_opt)[0]
        plt.plot(x, y_opt_actual, label="$y(u^*)$", alpha=0.5)
        plt.title("Optimal state " + model_name + " " + method_gradient.split(" ")[-1], fontsize="large")
        print("R2 of mean = ", 1. - np.mean( np.mean((y_opt.mean(axis=0) - y_opt_actual)**2)/y_opt_actual.var()) )
        print("R2 =", 1. - np.mean( np.mean((y_opt - y_opt_actual[None])**2, axis=(1,2))/y_opt_actual.var()), "+/-", np.std(1.- np.mean((y_opt - y_opt_actual[None])**2, axis=(1,2))/y_opt_actual.var()))
    else:
        plt.figure(figsize=figsize)
        plt.plot(x, y_opt.ravel(), label="opt. state")
        plt.title("Optimal state", fontsize="large")
    plt.plot(x, y_d.ravel(), linestyle="--", label="$y_d$")
    plt.xlabel("$x$"); plt.ylabel("$y$"); plt.ylim([0.95, 1.65])
    plt.legend(loc="lower right")
    plt.tight_layout()
    if savefigs:
        plt.savefig("Optimal_state_" + (model_name+"_")*(method_state!="conventional" and method_gradient!="conventional_adjoint") + method_state + "_" + method_gradient + ".pdf")
    plt.show()
    
    # plot optimal control
    plt.figure(figsize=figsize)
    plt.plot(x, u_opt.ravel(), label="$u^*$")
    if method_state=="NN":
        plt.title("Optimal control " + model_name + " " + method_gradient.split(" ")[-1], fontsize="large")
    else:
        plt.title("Optimal control", fontsize="large")
    plt.xlabel("$x$"); plt.ylabel("$u$")
    plt.legend()
    plt.tight_layout()
    if savefigs:
        plt.savefig("Optimal_control_" + (model_name+"_")*(method_state!="conventional" and method_gradient!="conventional_adjoint") + method_state + "_" + method_gradient + ".pdf")
    plt.show()
    
    # plot adjoint
    if method_gradient != "NN tangent":
        p_opt = problem.calculate_adjoint(y_opt).mean(axis=0)
        if method_gradient == "NN adjoint":
            # compare NN predicted state with numerical solution
            plt.figure(figsize=figsize)
            plt.plot(x, p_opt.ravel(), label="$\\tilde{p}(u^*)$")
            y_opt_actual = problem.calculate_state_conventional(u_opt)
            plt.plot(x, problem.calculate_adjoint_conventional(y_opt_actual).ravel(), color="red", label="$p(u^*)$")
            plt.title(model_name + ", " + method_state + " state, " + method_gradient, fontsize="large")
        else:
            plt.figure(figsize=figsize)
            plt.plot(x, p_opt.ravel(), label="adjoint p at opt. solution")
            plt.title(method_state + " state, " + method_gradient, fontsize="large")
        plt.xlabel("$x$"); plt.ylabel("$p$")#; plt.ylim([0.95, 1.65])
        plt.legend()
        plt.show()
    
    # plot cost history
    plt.figure(figsize=figsize)
    if method_state == "NN":
        plt.plot(np.arange(len(cost_history)), cost_history, label="$J(\\tilde{y},u)$")
        cost_history_true = []
        for u_ in u_history:
            y_ = problem.calculate_state_conventional(u_)
            cost_ = problem.cost(y_, u_)
            cost_history_true.append(cost_)
        plt.plot(np.arange(len(cost_history_true)), cost_history_true, linestyle="--", label="$J(y, u)$")
    else:
        plt.plot(np.arange(len(cost_history)), cost_history, label="$J(y,u)$")
    
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
    if method_state=="conventional" and method_gradient=="conventional adjoint":
        plt.title("Cost", fontsize="large")
    elif method_state=="NN" and method_gradient=="NN adjoint":
        plt.title("Cost {} adjoint".format(model_name), fontsize="large")
    elif method_state=="NN" and method_gradient=="NN tangent":
        plt.title("Cost {} tangent".format(model_name), fontsize="large")
    else:
        plt.title("Cost " + method_state + " state, " + method_gradient, fontsize="large")
    plt.yscale("log")
    plt.xlabel("Iteration"); plt.ylabel("$J$")
    plt.xticks(np.arange(4 + 1)*max_no_iters//4)
    plt.legend()
    plt.tight_layout()
    if savefigs:
        plt.savefig("Cost_history_" + (model_name+"_")*(method_state!="conventional" and method_gradient!="conventional_adjoint") + method_state + "_" + method_gradient + ".pdf")
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