# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:33:22 2024

@author: L390
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:20:17 2024

Generates data (u_i, y_i) for exponential decay problem y' = -y + u
"""

import numpy as np
import torch
from crank_nicolson_heat_eq import crank_nicolson


def solve_state_eq(u, y_IC, y_BCs, D, t_span, x_span, batched=True):
    # if batched=False, u has shape (N_t, N_x,...)
    # solve PDE y_t = D*y_xx + u
    t0, tf = t_span
    x0, fx = x_span
    if not batched:
        u = u[None,:]
    
    y = crank_nicolson(y_IC, y_BCs, D, u, t0=0., tf=1., x0=0.,xf=1)
    if not batched:
        return y[0]
    else:
        return y
    
    
def solve_adjoint_eq(y, y_d, p_TC, p_BCs, D, t_span, x_span, batched=True):
    # if batched=False, y, y_d have shape (N_t, N_x,...)
    # solve ODE -p_t = -Dp_xx + (y-y_d)
    # backwards in time: p(t-1) = p(t) - dt*p' = p(t) + dt (-p')
    # use the fact that this is equivalent with solving the state equation
    p = solve_state_eq(y-y_d, p_TC, p_BCs, D, t_span, x_span, batched=batched)
    p = np.ascontiguousarray(np.flip(p, axis=batched)) # flip time axis and store as new array
    return p



def generate_controls(t, x,
                      n_samples,
                      n_t_coeffs,
                      n_x_coeffs,
                      normalize=True,
                      u_max=10.):
    # generate controls u(t,x) as a superposition of products sin(omega*t)sin(k*x), cos(omega*t)*sin(k*x)
    # where k = n*pi/L, n = 1,2,3,...,n_x_coeffs
    # s.t. u(t,0) = u(t,L) = 0
    # omega = pi n/T, n = 0,1,2,3,...,n_t_coeffs-1
    T = t[-1] - t[0]
    t_cos_basis = torch.stack([ torch.cos(i*torch.pi/T * t) for i in range(n_t_coeffs)]) # includes cos(0*t) = 1
    t_sin_basis = torch.stack([ torch.sin(i*torch.pi/T * t) for i in range(n_t_coeffs)]) # includes sin(0*t) = 0
    t_coeffs = 2 *torch.rand(size=(n_samples, 2*n_t_coeffs)) - 1. # coeffs for cos and sin
    t_cos_terms = torch.einsum('jt, ij -> ijt', t_cos_basis, t_coeffs[:,:n_t_coeffs])
    t_sin_terms = torch.einsum('jt, ij -> ijt', t_sin_basis, t_coeffs[:,n_t_coeffs:])
    
    L = x[-1]-x[0]
    x_sin_basis = torch.stack([ torch.sin(i*torch.pi/L * x) for i in range(1, n_x_coeffs+1)]) # *includes* wavenumber k = n+1
    x_coeffs = 2* torch.rand(size=(n_samples,n_x_coeffs)) - 1.
    x_sin_terms = torch.einsum('jx, ij -> ijx', x_sin_basis, x_coeffs)
    
    u = torch.einsum('ijt, ikx -> itx', t_cos_terms + t_sin_terms, x_sin_terms)
    
    if normalize:
        # set max amplitude of control u to be in interval (1, u_max)
        control_strengths = (u_max - 1.)*torch.rand(size=(n_samples,1,1)) + 1.
        u = control_strengths * (u/torch.max(u.flatten(start_dim=1).abs()))
    return u

def generate_data(N_t,
                  N_x,
                  t_span,
                  x_span,
                  IC=None,
                  BCs=None,
                  n_t_coeffs=5,
                  n_x_coeffs=5,
                  n_samples=1024,
                  u_max=10.,
                  diffusion_coeff=0.25,
                  generate_adjoint=False,
                  y_d=None, 
                  seed=None,
                  add_noise=False):
    """
    basis (str) : which basis to use for P_n. One of "monomial", "Chebyshev", "Legendre", "Bernstein"
    """
    # set up data array
    data = {}
    
    # domain
    t0, tf = t_span
    x0, xf = x_span
    t = torch.linspace(0.,1.,N_t).view(N_t)
    x = torch.linspace(0.,1.,N_x).view(N_x)
    
    # seed RNG
    if seed:
        torch.manual_seed(seed)
    
    if add_noise:
        noise = 1e-2*u_max*torch.randn(size=(n_samples, N_t, N_x, 1), dtype=torch.float32)
    
    if not generate_adjoint:
        if IC is None:
            IC = np.sin(np.pi/(xf-x0)*x)
        if BCs is None:
            BCs = (np.zeros_like(t), np.zeros_like(t))
            
        # generate uniformly sampled u coefficients then calculate the state using numerical integration
        u = generate_controls(t, x, n_samples, n_t_coeffs, n_x_coeffs, u_max=u_max)
        y = torch.tensor( solve_state_eq(u.numpy(), IC, BCs, diffusion_coeff, t_span, x_span), dtype=torch.float32 )
        if add_noise:
            y += noise
        # return u and y as (n_samples, N_t*N_x) shaped arrays
        data["u"] = u
        data["tx"] = torch.cartesian_prod(t, x)[None].repeat(n_samples,1,1)
        data["y"] = y
        return data
    
    else:
        if IC is None:
            IC = np.sin(np.pi/(xf-x0)*x)
        if BCs is None:
            BCs = (np.zeros_like(t), np.zeros_like(t))
        # generates adjoint p by sampling y uniformly and solving adjoint eq for p
        y = generate_controls(t, x, n_samples, n_t_coeffs, n_x_coeffs, u_max=u_max)
        p = torch.tensor( solve_adjoint_eq(y.numpy(), y_d, IC, BCs, diffusion_coeff, t_span, x_span), dtype=torch.float32 )
        if add_noise:
            p += noise
        data["y-y_d"] = y-y_d
        data["tx"] = torch.cartesian_prod(t, x)[None].repeat(n_samples,1,1)
        data["p"] = p
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
    tx = data["tx"]
    
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
    data["tx"] = torch.cat((tx, tx[0].repeat(n_augmented_samples,1,1)))