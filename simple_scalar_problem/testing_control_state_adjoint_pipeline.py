# -*- coding: utf-8 -*-
"""
Comparing the improved Forward Euler solver and the neural operators
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from generate_data import solve_state_eq, solve_adjoint_eq
plt.style.use("ggplot")

savefigs = True

yd = 1.5
y0 = 1.
N = 128
x = np.linspace(0.,1.,N)

p = (yd - y0)*(np.exp(2*x) - np.exp(x + 1.))

y = y0*np.exp(2*x) - yd*(np.exp(2*x) - 1.)               

u = 3*y0*np.exp(2*x) - yd*(3*np.exp(2*x) - 1.)

y_FE = solve_state_eq(u[None])[0]

p_FE = solve_adjoint_eq(y_FE[None], yd)[0]

fig = plt.figure(figsize=(5,3.5))
plt.plot(x, u); plt.xlabel("$x$"); plt.ylabel("$u$")
plt.title("Control")
plt.tight_layout()
if savefigs:
    plt.savefig("control_to_adjoint_pipeline_u.pdf")
plt.show()

fig, axs = plt.subplots(figsize=(8,4), ncols=2)

axs[0].plot(x, y, linestyle="--", label="analytical")
axs[0].plot(x, y_FE, label="numerical")
axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$y$")
axs[0].set_title("State")
axs[0].legend()

axs[1].plot(x, p, linestyle="--", label="analytical")
axs[1].plot(x, p_FE, label="numerical")
axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$p$")
axs[1].set_title("Adjoint")
axs[1].legend()

fig.tight_layout()
if savefigs:
    plt.savefig("control_to_adjoint_pipeline_numerical.pdf")

plt.show()

# plotting neural operator predictions

import glob
import pickle

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parents[0]
sys.path.append(str(root))
sys.path.append(str(root / "utils"))
sys.path.append(str(root / "simple_scalar_problem"))

import FNO
import DeepONet
for model_name in ["FNO", "DON"]:
    print(model_name)
    def predict(x, u, model_list):
        u_tens = torch.tensor(u, dtype=torch.float32)[None]
        x_tens = torch.tensor(x, dtype=torch.float32)[None]
        if model_name=="FNO":
            pred = torch.cat([model(u_tens[...,None]) for model in model_list])
        else:
            pred = torch.cat([model(u_tens, x_tens[...,None]) for model in model_list])
        return pred.mean(axis=0).detach().numpy(), pred.std(axis=0).detach().numpy()
    
    # load models
    models = dict()
    state_filename = glob.glob('.\\state_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]
    with open(state_filename, 'rb') as infile:
        models["state"] = pickle.load(infile)[:]
    
    adjoint_filename = glob.glob('.\\adjoint_experiments_{}_15_models\\models_list*.pkl'.format(model_name))[0]
    with open(adjoint_filename, 'rb') as infile:
        models["adjoint"] = pickle.load(infile)[:]
    
    y_pred, y_sd = predict(x, u, models["state"])
    print(f"Mean std.dev. of states {y_sd.mean() = }, std.dev. of mean state {y_pred.std() = }")
    
    p_pred, p_sd = predict(x, y-yd, models["adjoint"])
    print(f"Mean std.dev. of adjoints {p_sd.mean() = }, std.dev. of mean adjoint {p_pred.std() = }")
    
    fig, axs = plt.subplots(figsize=(8,4), ncols=2)
    #axs[0].plot(x, u); axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$u$")
    #axs[0].set_title("Control")

    axs[0].plot(x, y, linestyle="--", label="analytical")
    axs[0].plot(x, y_pred, label=model_name, color="red")
    axs[0].plot(x, y_pred + y_sd, color="red", alpha=0.4)
    axs[0].plot(x, y_pred - y_sd, color="red", alpha=0.4)
    axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$y$")
    axs[0].set_title("State")
    axs[0].legend()

    axs[1].plot(x, p, linestyle="--", label="analytical")
    axs[1].plot(x, p_pred, label=model_name, color="red")
    axs[1].plot(x, p_pred + p_sd, color="red", alpha=0.4)
    axs[1].plot(x, p_pred - p_sd, color="red", alpha=0.4)
    axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$p$")
    axs[1].set_title("Adjoint")
    axs[1].legend()

    fig.tight_layout()
    if savefigs:
        plt.savefig("control_to_adjoint_pipeline_{}.pdf".format(model_name))
    plt.show()