# -*- coding: utf-8 -*-
import numpy as np
import torch

import os
import glob
import pickle

from FNO import FNO
from DeepONet import DeepONet
from generate_data import solve_state_eq, solve_adjoint_eq

from OC_simple_scalar_problem import OC_problem
from utils.bootstrap_OC import *

# create data directory to store results
data_dir_name = 'bootstrap_results'
problem_dir_name = "simple_scalar_problem"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, problem_dir_name, data_dir_name)
if not os.path.exists(data_dir):
        os.makedirs(data_dir)

# set parameters
N = 128
nu = 1e-2

model_name = "DON" # NN model to use: FNO or DON

state_filename = glob.glob('.\\state_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]

adjoint_filename = glob.glob('.\\adjoint_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]

NO_problem = OC_problem("NN", "NN adjoint", nu, None, None, model_name)

u0 = np.zeros(shape=(1,N,1))

max_no_iters = 40 # max. no. of optimisation iterations
B = 1000 # number of boostraps
n_models_max = 10

cost_stats, cost_true_stats, u_opt_stats = bootstrap_OC(state_filename, adjoint_filename, NO_problem, u0, max_no_iters=max_no_iters, n_models_max=n_models_max, B=B)

filename_cost = "cost_stats.pkl"
with open(os.path.join(data_dir, filename_cost), "wb") as outfile:
    pickle.dump(cost_stats, outfile)
filename_cost_true = "cost_true_stats.pkl"
with open(os.path.join(data_dir, filename_cost_true), "wb") as outfile:
    pickle.dump(cost_true_stats, outfile)
filename_u_opt = "u_opt_stats.pkl"
with open(os.path.join(data_dir, filename_u_opt), "wb") as outfile:
    pickle.dump(u_opt_stats, outfile)