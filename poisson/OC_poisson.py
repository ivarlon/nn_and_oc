# -*- coding: utf-8 -*-
"""
controlling POISSON
J = 0.5*||y-y_d||^2 + 0.5*nu*||u||^2
subject to  $\nabla^2 y = - u$

opt. conditions:
    $\nabla^2 p = -(y-y_d)
    0 = grad(J) = nu*u - p
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import torch

from torch.func import stack_module_state, functional_call, vmap # to vectorise ensemble
import copy

import glob
import pickle

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parents[0]
sys.path.append(str(root))
sys.path.append(str(root / "utils"))
sys.path.append(str(root / "heat_equation"))
sys.path.append(str(root / "poisson"))
from FNO import FNO
from DeepONet import DeepONet
from generate_data_poisson import solve_poisson
from optimization_routines import grad_descent, conjugate_gradient, linesearch

N = 64 # no. of spatial points

L = 1.
x1 = np.linspace(0.,L, N)
x2 = np.linspace(0.,L, N)
h = L/(N-1)

X1, X2 = np.meshgrid(x1,x2, indexing='ij')

nu = 5e-5 # penalty on control u


# desired state for OC is single peak
y_d = 0.25*np.sin(2*np.pi*X1)**2 * np.sin(2*np.pi*X2)**2

# boundary conditions on state
BCs = [np.zeros(N) for i in range(4)] # Dirichlet boundary conditions on state

# boundary conditions on adjoint
p_BCs = [np.zeros(N) for i in range(4)] # zero Dirichlet boundary conditions

def load_models(filename):
    with open(filename, 'rb') as infile:
        models_list = pickle.load(infile)
    return models_list

def get_R2_of_prediction_history(u_history, model_ensemble, numerical_solver):
    ensemble_pred_R2 = []
    mean_R2 = [] # mean R2 of intra-ensemble preds
    std_R2 = [] # std.dev. of R2 of intra-ensemble preds
    for i in range(len(u_history)):
        u = u_history[i]
        preds = model_ensemble(u)
        target = numerical_solver(u)
        ensemble_pred_R2.append(1. - np.mean( (preds.mean(axis=0) - target[0])**2, axis=(-2,-1))/target.var())
        mean_R2.append(1. - np.mean( np.mean( (preds - target)**2, axis=(-2,-1))/target.var() ) )
        std_R2.append(np.std(1. - np.mean( (preds - target)**2, axis=(-2,-1))/target.var() ) )
    return np.array(ensemble_pred_R2), np.array(mean_R2), np.array(std_R2)


class OC_problem:
    # This class defines functions to be used in the OC problem
    def __init__(self, method_state, method_gradient, nu, models_list=None, model_name=None):
        """
        method_state (str) : method by which to calculate state y
                either 'conventional' or 'NN'
        method_gradient (str) : method by which the gradient dJ/du is computed
                either 'conventional adjoint', 'NN adjoint', 'NN tangent' (the latter doesn't calculate adjoint state)
        """
        # penalty on control
        self.nu = nu
        
        # initialise models
        self.initialise_ensemble(models_list)
        
        # state methods
        if method_state == "conventional":
            self.calculate_state = self.calculate_state_conventional
        
        elif method_state == "NN":
            if model_name == "FNO":
                self.calculate_state = self.calculate_state_FNO
            
            elif model_name == "DON":
                self.r = torch.cartesian_prod(torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32))[None]
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
            # load models
            self.gradient_cost = self.gradient_cost_adjoint_method
            if model_name == "FNO":
                self.calculate_adjoint = self.calculate_adjoint_FNO
            elif model_name == "DON":
                self.r = torch.cartesian_prod(torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32))[None]
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
        return 0.5*h**2*np.sum((y-y_d)**2) + 0.5*self.nu*h**2*np.sum(u**2)
    
    def reduced_cost(self, u):
        y = self.calculate_state(u).mean(axis=0)[None]
        return self.cost(y, u)
    
    def initialise_ensemble(self, models):
        
        if models is not None:
            self.params, self.buffers = stack_module_state(models)
            self.base_model = copy.deepcopy(models[0]).to("meta") # make a "meta" version of models
    
    ###########################################
    # methods to calculate state and gradient #
    ###########################################
    def calculate_state_conventional(self, u):
        # solves system of equations
        return solve_poisson(u, BCs)
    
    def calculate_state_FNO(self, u):
        # calculate ensemble predictions
        u = torch.tensor(u, dtype=torch.float32).unsqueeze(-1)
        def state_ensemble(params, buffers, u):
            return functional_call( self.base_model, (params, buffers), (u,))
        
        y = vmap(state_ensemble, in_dims=(0, 0, None))(self.params, self.buffers, u)
        y = y.view(len(y),N,N)
        y = y.detach().numpy()
        return y
    
    def calculate_state_DON(self, u):
        u = torch.tensor(u, dtype=torch.float32)
        # calculate ensemble predictions
        def state_ensemble(params, buffers, u):
            return functional_call( self.base_model, (params, buffers), (u, self.r))
        
        y = vmap(state_ensemble, in_dims=(0, 0, None))(self.params, self.buffers, u)
        y = y.view(len(y),N,N)
        y = y.detach().numpy()
        return y
    
    def calculate_adjoint_conventional(self, y):
        return solve_poisson(y-y_d, p_BCs)
    
    def calculate_adjoint_FNO(self, y):
        scale = 100.
        y_y_d = scale*torch.tensor(y-y_d, dtype=torch.float32).unsqueeze(-1)
        # calculate ensemble predictions
        def ensemble(params, buffers, y):
            return functional_call( self.base_model, (params, buffers), (y,))
        
        p = vmap(ensemble, in_dims=(0, 0, None))(self.params, self.buffers, y_y_d)
        p = p/scale
        p = p.view(len(p)*len(y_y_d),N,N)
        p = p.detach().numpy()
        return p
    
    def calculate_adjoint_DON(self, y):
        scale = 100.
        y_y_d = scale*torch.tensor(y-y_d, dtype=torch.float32)
        # calculate ensemble predictions
        def ensemble(params, buffers, y):
            return functional_call( self.base_model, (params, buffers), (y, self.r))
        
        p = vmap(ensemble, in_dims=(0, 0, None))(self.params, self.buffers, y_y_d)
        p = p/scale
        p = p.view(len(p)*len(y_y_d),N,N)
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
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True).unsqueeze(-1)
        
        def ensemble(params, buffers, u):
            return functional_call( self.base_model, (params, buffers), (u,))
        
        # Calculate vJp for dNN/du^T (y-y_d)
        calculate_state_vjp = lambda u: vmap(ensemble, in_dims=(0, 0, None))(self.params, self.buffers, u).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state_vjp, u)
        y_y_d = y-torch.tensor(y_d, dtype=torch.float32).flatten().unsqueeze(-1)
        dJdy_times_dydu = grad_u(y_y_d)[0].view(u_np.shape).detach().numpy()
        return J_u + dJdy_times_dydu
    
    def DON_tangent(self, u):
        # Calculates gradient of cost as dJ = J_u + J_y dy/du
        # J_u = nu*u
        # dy/du J_y = grad(NN;u)*(y-y_d)
        
        u_np = u
        J_u = self.nu*u_np
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        
        def ensemble(params, buffers, u):
            return functional_call( self.base_model, (params, buffers), (u, self.r))
        
        # Calculate vJp for dNN/du^T (y-y_d)
        calculate_state_vjp = lambda u: vmap(ensemble, in_dims=(0, 0, None))(self.params, self.buffers, u).mean(axis=0)
        y, grad_u = torch.func.vjp(calculate_state_vjp, u)
        y_y_d = y-torch.tensor(y_d).flatten().unsqueeze(-1)
        dJdy_times_dydu = grad_u(y_y_d)[0].view(u_np.shape).detach().numpy()
        
        return J_u + dJdy_times_dydu

if __name__=="__main__":
    savefigs = False
    figsize=(3.5,3)
    fontsize="large"
    
    model_name = "FNO" # "DON" # NN model to use (if any)
    
    method_state = "NN"
    method_gradient = "NN tangent"
    
    #models_filename = glob.glob('.\state_experiments_{}_1407070\models_list_*.001.pkl'.format(model_name))[-1]
    models_filename = glob.glob('.\{}_models\models_list*.001.pkl'.format(model_name))[-1]
    models_list = load_models(models_filename)[:10]
    if "NN" in [method_state, method_gradient.split(" ")[0]]:
        print("Using n={} models.\n".format(len(models_list)))
    #models_list = []
    #for i in range(len(m_list)):
    #    if i in [0,3,4,5,9]:
    #        models_list.append(m_list[i])
    problem = OC_problem(method_state, method_gradient, nu, models_list, model_name)
    
    
    #====================
    # Do optimal control
    #====================
        
    u0 = 1./(2*np.pi*0.2**2)*np.exp(-0.5*((X1-0.5)**2 + (X2-0.5)**2)/0.2**2)[None] # initial guess is zero
    u0 *= 80./np.abs(u0).max()
    max_no_iters = 300 # max. no. of optimisation iterations
    
    
    # time optimisation routine
    import time
    time_start = time.time()
    
    u_history, cost_history, grad_history  = conjugate_gradient(problem.reduced_cost,
                                             problem.gradient_cost,
                                             u0,
                                             max_no_iters=max_no_iters,
                                             return_full_history=True,
                                             print_every=5)
    
    print()
    print("Optimisation took", round(time.time() - time_start, 1), "secs")
    
    u_opt = u_history[-1]
    
    # plot optimal state
    y_opt = problem.calculate_state(u_opt)#.mean(axis=0)
    fig = plt.figure(figsize=(2*figsize[0],figsize[1]))
    ax0 = fig.add_subplot(121)
    contour0 = ax0.contourf(X1, X2, y_opt.mean(axis=0), levels=np.linspace(y_opt.mean(axis=0).min(), y_opt.mean(axis=0).max()))
    fig.colorbar(contour0, ax=ax0, label="$y$")
    ax0.set_xlabel("$x_1$"); ax0.set_ylabel("$x_2$")
    
    
    ax1 = fig.add_subplot(122)
    if method_gradient.split(" ")[0]=="NN":
        ax0.set_title("Optimal state, " + model_name  + " " + method_gradient.split(" ")[-1], fontsize=fontsize)
        
        y_opt_actual = problem.calculate_state_conventional(u_opt).mean(axis=0)
        #rel_error = np.abs((y_opt - y_opt_actual)).mean(axis=0)#/(y_opt_actual + 1e-7))
        #contour1 = ax1.contourf(X1, X2, rel_error**0.5, levels=np.linspace((rel_error[rel_error<1e2]**0.5).min(), (rel_error[rel_error<1e2]**0.5).max()))
        contour1 = ax1.contourf(X1, X2, y_opt_actual, levels=np.linspace(y_opt_actual.min(), y_opt_actual.max()))
        fig.colorbar(contour1, ax=ax1, label="$y$")
        ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
        ax1.set_title("Numerical solution", fontsize=fontsize)
        print("R2 of mean = ", 1. - np.mean( np.mean((y_opt.mean(axis=0) - y_opt_actual)**2)/y_opt_actual.var()) )
        print("R2 =", 1. - np.mean( np.mean((y_opt - y_opt_actual[None])**2, axis=(1,2))/y_opt_actual.var()), "+/-", np.std(1.- np.mean((y_opt - y_opt_actual[None])**2, axis=(1,2))/y_opt_actual.var()))
    else:
        ax0.set_title("Optimal state")
        
        contour1 = ax1.contourf(X1, X2, y_d, levels=np.linspace(y_d.min(), y_d.max()))
        fig.colorbar(contour1, ax=ax1, label="$y_d$")
        ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
        ax1.set_title("Desired state $y_d$", fontsize=fontsize)
    fig.tight_layout() 
    if savefigs:
        plt.savefig("Optimal_state_" + (model_name+"_")*(method_state!="conventional" and method_gradient!="conventional_adjoint") + method_state + "_" + method_gradient + ".pdf")
    plt.show()
    """
    if method_state=="NN":
        fig, axs = plt.subplots(ncols=len(y_opt), figsize=(18,6))
        from matplotlib.ticker import FormatStrFormatter
        contours = []
        for i in range(len(y_opt)):
            contours.append(axs[i].contourf(tt, xx, y_opt[i], levels=np.linspace(y_opt[i].min(), y_opt[i].max())))
            cbar = fig.colorbar(contours[i], ax=axs[i])
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.tight_layout()
        plt.show()
    """
    # plot optimal control
    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(X1, X2, u_opt[0], levels=np.linspace(u_opt.min(), u_opt.max()))
    fig.colorbar(contour, ax=ax, label="$u$")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    if method_gradient.split(" ")[0]=="NN":
        ax.set_title("Optimal control, " + model_name + " " + method_gradient.split(" ")[-1], fontsize="medium")
    else:
        ax.set_title("Optimal control", fontsize=fontsize)
    fig.tight_layout()
    if savefigs:
        plt.savefig("Optimal_control_" + (model_name+"_")*(method_state!="conventional" and method_gradient!="conventional_adjoint") + method_state + "_" + method_gradient + ".pdf")
    plt.show()
    
    # plot cost history
    fig = plt.figure(figsize=figsize)
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
    plt.xlabel("Iteration"); plt.ylabel("$J$"); plt.yscale("log")
    if method_state=="conventional" and method_gradient=="conventional adjoint":
        plt.title("Cost", fontsize="large")
    elif method_state=="NN" and method_gradient=="NN adjoint":
        plt.title("Cost: {} adjoint".format(model_name), fontsize=fontsize)
    elif method_state=="NN" and method_gradient=="NN tangent":
        plt.title("Cost: {} tangent".format(model_name), fontsize=fontsize)
    else:
        plt.title("Cost: " + method_state + " state, " + method_gradient, fontsize=fontsize)
    plt.legend()
    plt.tight_layout()
    if savefigs:
        plt.savefig("Cost_history_" + (model_name+"_")*(method_state!="conventional" and method_gradient!="conventional_adjoint") + method_state + "_" + method_gradient + ".pdf")
    plt.show()
    
    if method_state=="NN":
        fig = plt.figure(figsize=figsize)
        R2_ensemble, R2_mean, R2_std = get_R2_of_prediction_history(u_history, problem.calculate_state, problem.calculate_state_conventional)
        plt.plot(np.arange(max_no_iters + 1), R2_ensemble, label="Ensemble")
        plt.plot(np.arange(max_no_iters + 1), R2_mean, color="black", label="Intra-ensemble")
        plt.plot(np.arange(max_no_iters + 1), R2_mean + R2_std, color="black", linestyle="--")
        plt.plot(np.arange(max_no_iters + 1), R2_mean - R2_std, color="black", linestyle="--")
        plt.xlabel("Iteration"); plt.ylabel("$R^2$")
        plt.ylim([0., 1.1])
        plt.legend(loc="lower left")
        plt.title("$R^2$ of prediction history", fontsize=fontsize)
        plt.tight_layout()
        if savefigs:
            plt.savefig("R2_" + (model_name+"_")*(method_state!="conventional" and method_gradient!="conventional_adjoint") + method_state + "_" + method_gradient + ".pdf")
        plt.show()