# -*- coding: utf-8 -*-
"""
Performs a bootstrap of OC using an ensemble of neural operators.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

import glob
import pickle

#import sys
#from pathlib import Path
#root = Path(__file__).resolve().parent.parents[0]
#sys.path.append(str(root))
#sys.path.append(str(root / "utils"))
#sys.path.append(str(root / "heat_equation"))
from FNO import FNO
from DeepONet import DeepONet
#from generate_data_heat_eq import solve_state_eq, solve_adjoint_eq
from optimization_routines import grad_descent, conjugate_gradient, linesearch
#from OC_heat_eq import *
from tqdm import trange

torch.manual_seed(31416)
def bootstrap_OC(state_models_filename, adjoint_models_filename, problem, u0, max_no_iters, n_models_max=None, B=1000):
    """
    Performs a bootstrap for n=1,...,n_models_max to get statistics for cost (mean+stddev) when doing OC with neural operators.
    state_models_filename (str): filename that contains the models
    n_models_max (int): max number of models to compute bootstrap for
    B (int): number of bootstraps to perform for each n
    problem (OC_problem): instance of class OC_problem defined in main OC files
    """
    
    with open(state_models_filename, 'rb') as infile:
        state_models_list = pickle.load(infile)
    
    with open(adjoint_models_filename, 'rb') as infile:
        adjoint_models_list = pickle.load(infile)
    
    if n_models_max is None:
        n_models_max = len(state_models_list) - 1
    
    u_opt_stats = dict(mean = [], sd = [])
    cost_stats = dict(mean = [], sd = [])
    cost_true_stats = dict(mean = [], sd = [])
    for n_models in range(1, n_models_max+1):
        print("{:g}/{:g}".format(n_models, n_models_max))
        cost_list = []
        cost_true_list = []
        u_opt_list = []
        
        for b in trange(B):
            
            # select n_models models randomly
            state_models_idx = torch.randperm(len(state_models_list))[:n_models]
            adjoint_models_idx = torch.randperm(len(adjoint_models_list))[:n_models]
            
            sampled_state_models = [state_models_list[i] for i in state_models_idx]
            sampled_adjoint_models = [adjoint_models_list[i] for i in adjoint_models_idx]
            
            problem.models["state"] = sampled_state_models
            problem.models["adjoint"] = sampled_adjoint_models
            
            u_opt, cost_history, grad_history = conjugate_gradient(problem.reduced_cost,
                                                     problem.gradient_cost,
                                                     u0,
                                                     max_no_iters=max_no_iters)
            cost_list.append(cost_history[-1])
            cost_true = problem.cost(problem.calculate_state_conventional(u_opt), u_opt)
            cost_true_list.append(cost_true)
            u_opt_list.append(u_opt)
            
        cost_list = np.array(cost_list)
        cost_stats["mean"].append( cost_list.mean() )
        cost_stats["sd"].append( cost_list.std() )
        
        cost_true_list = np.array(cost_true_list)
        cost_true_stats["mean"].append( cost_true_list.mean() )
        cost_true_stats["sd"].append( cost_true_list.std() )
        
        u_opt_list = np.concatenate(u_opt_list)
        u_opt_stats["mean"].append( u_opt_list.mean() )
        u_opt_stats["sd"].append( u_opt_list.mean(axis=tuple(range(1,u_opt_list.ndim))).std() )
        
    return cost_stats, cost_true_stats, u_opt_stats