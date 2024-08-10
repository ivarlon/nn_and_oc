# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:20:17 2024

Generates data (u_i, y_i) for exponential decay problem y' = -y + u
"""

import numpy as np
import torch

def improved_forward_euler(fun, y0, n_points, x_span, u):
    
    n_batches = u.shape[0]
    
    # set up domain array
    x0, xf = x_span
    x = np.linspace(x0, xf, n_points)
    
    # set up solution array
    y = np.zeros(shape=(n_batches, n_points))
    y[:,0] = y0
    
    # step length
    h = (xf-x0)/(n_points-1)
    
    # integration loop
    for i in range(n_points-1):
        k1 = fun(x[i], y[:,i]) + u[:,i]
        pred = y[:,i] + h*k1
        k2 = fun(x[i+1], pred) + u[:,i+1]
        y[:,i+1] = y[:,i] + h/2 * (k1 + k2)
    
    return y

def solve_state_eq(u, y0=1., batched=True):
    # if batched=False, u has shape (N,...)
    # solve ODE y' = -y + u
    state_eq = lambda x, y: -y
    if not batched:
        u = u[None]
    N = u.shape[1]
    y = improved_forward_euler(state_eq, y0, n_points=N, x_span=(0.,1.), u=u)
    if not batched:
        return y[0]
    else:
        return y
    
    
def solve_adjoint_eq(y, y_d, pf=0., batched=True):
    # if batched=False, y, y_d have shape (N,...)
    # solve ODE -p' = -p + (y-y_d)
    # backwards in time: p(t-1) = p(t) - dt*p'
    adjoint_eq = lambda x, p: -p
    if not batched:
        y = y[None]
        y_d = y_d[None]
    else:
        y_d = y_d[None]
    N = y.shape[1]
    p = improved_forward_euler(adjoint_eq, [pf], n_points=N, x_span=(0.,1.), u=np.flip(y-y_d))
    p = np.ascontiguousarray( np.flip(p, axis=1) )
    if not batched:
        return p[0]
    else:
        return p

def generate_controls(x,
                      basis,
                      n_samples,
                      coeff_range,
                      n_coeffs):
    if basis=="monomial":
        polynomial = lambda coeffs, x: np.polynomial.Polynomial(coeffs, domain=[0.,1.])(x)
    elif basis=="Chebyshev":
        polynomial = lambda coeffs, x: np.polynomial.Chebyshev(coeffs, domain=[0.,1.])(x)
    elif basis=="Legendre":
        # possible to scale Legendre polynomial i by sqrt(2i + 1) to get normalised: torch.sqrt(2*torch.arange(n_coeffs))*coeffs
        polynomial = lambda coeffs, x: np.polynomial.Legendre(coeffs, domain=[0.,1.])(x)
    else:
        print("Enter a valid polynomial basis")
        return None
    u_coeffs = 2*coeff_range*torch.rand(size=(n_samples,n_coeffs)) - coeff_range
    u = torch.stack( [torch.tensor(polynomial(coeffs, x),dtype=torch.float32) for coeffs in u_coeffs] )
    return u

def generate_data(N,
                  basis,
                  n_samples=1024,
                  sample_input_function_uniformly=True,
                  generate_adjoint=False,
                  y_d = None, 
                  coeff_range=3., 
                  n_coeffs=8, 
                  boundary_condition=1., 
                  seed=None,
                  add_noise=False):
    """
    basis (str) : which basis to use for P_n. One of "monomial", "Chebyshev", "Legendre", "Bernstein"
    """
    # set up data array
    
    data = {}
    x = torch.linspace(0.,1.,N)
    
    # seed RNG
    if seed:
        torch.manual_seed(seed)
    
    if add_noise:
        noise = 1e-2*coeff_range*torch.randn(size=(n_samples, N), dtype=torch.float32)
    
    if not generate_adjoint:
        # generate uniformly sampled u coefficients then calculate the state using numerical integration
        u = generate_controls(x, basis, n_samples, coeff_range, n_coeffs)
        y = torch.tensor( solve_state_eq(u.numpy(),boundary_condition), dtype=torch.float32 )
        if add_noise:
            noise = 1e-2*coeff_range*torch.randn(size=(n_samples, N), dtype=torch.float32)
            y += noise
        data["u"] = u
        data["x"] = x.view(N,1).repeat(n_samples,1,1) # add batch axis 0 and axis 2 for dim(x) (=1)
        data["y"] = y.unsqueeze(-1) # add final singleton axis to match x.shape
        return data
    
    else:
        # generates adjoint p by sampling y uniformly
        y = generate_controls(x, basis, n_samples, coeff_range, n_coeffs)
        p = torch.tensor( solve_adjoint_eq(y.numpy(), y_d=y_d.numpy(), pf=boundary_condition), dtype=torch.float32 )
        if add_noise:
            noise = 1e-2*coeff_range*torch.randn(size=(n_samples, N), dtype=torch.float32)
            p += noise
        data["y-y_d"] = y - y_d
        data["x"] = x.view(N).repeat(n_samples,1,1)
        data["p"] = p.unsqueeze(-1) # add final singleton axis to match x.shape
        return data
    

def augment_data(data, n_augmented_samples, n_combinations, max_coeff, adjoint=False):
    # create linear combinations of solutions to get new solutions
    # returns n_augmented_samples of these lin.combs.
    # n_combinations (int) : number of solutions to combine
    # max_coeff (int) : max absolute size of lin.comb. coeffs
    # adjoint (bool) : must be True if augmenting adjoint data, False if state data
    
    if not adjoint:
        u = data["u"]
        y = data["y"]
        n_samples = y.shape[0]
    else:
        y_y_d = data["y-y_d"]
        p = data["p"]
        n_samples = p.shape[0]
    x = data["x"]
    
    assert n_samples > n_combinations, "number of samples must be greater than number of combinations"
    
    # index array for which samples to combine
    idx = torch.stack([ torch.randperm(n_samples)[:n_combinations] for i in range(n_augmented_samples) ])
    
    # sample lin.comb. coefficients from
    coeffs = 2*max_coeff*torch.rand(size=(n_augmented_samples,n_combinations)) - max_coeff
    
    # normalise sum of each coeff vector to 1 (to ensure lin. comb. respects boundary conditions)
    coeffs = coeffs/(coeffs.sum(dim=1, keepdims=True) + 1e-6)
    if not adjoint:
        y_aug = torch.einsum('nc..., nc...->n...', coeffs, y[idx] )
        u_aug = torch.einsum('nc..., nc...->n...', coeffs, u[idx] )
        data["u"] = torch.cat((u, u_aug))
        data["y"] = torch.cat((y, y_aug))
    else:
        p_aug = torch.einsum('nc..., nc...->n...', coeffs, p[idx] )
        y_y_d_aug = torch.einsum('nc..., nc...->n...', coeffs, y_y_d[idx] )
        data["y-y_d"] = torch.cat((y_y_d, y_y_d_aug))
        data["p"] = torch.cat((p, p_aug))
    
    data["x"] = torch.cat((x, x[0].repeat(n_augmented_samples,1,1)))