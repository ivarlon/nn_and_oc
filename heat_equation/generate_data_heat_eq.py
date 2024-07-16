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

torch.manual_seed(10)

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
    
    
def solve_adjoint_eq(y, y_d, pf=0., batched=True):
    # if batched=False, y, y_d have shape (N,...)
    # solve ODE -p' = -p + (y-y_d)
    # backwards in time: p(t-1) = p(t) - dt*p' = p(t) + dt (-p')
    adjoint_eq = lambda x, p: -p
    if not batched:
        y = y[None,:]
        y_d = y_d[None,:]
    N = y.shape[1]
    p = improved_forward_euler(adjoint_eq, [pf], n_points=N, x_span=(0.,1.), u=np.flip(y-y_d))
    p = np.flip(p, axis=1)
    if not batched:
        return p[0]
    else:
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
                  y_IC=None,
                  y_BCs=None,
                  n_t_coeffs=5,
                  n_x_coeffs=5,
                  n_samples=1024,
                  u_max=10.,
                  diffusion_coeff=0.5,
                  sample_input_function_uniformly=True,
                  generate_adjoint=False,
                  y_d = None,
                  boundary_condition=1., 
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
    
    if sample_input_function_uniformly:
        
        if not generate_adjoint:
            if y_IC is None:
                y_IC = np.sin(np.pi/(xf-x0)*x)
            if y_BCs is None:
                y_BCs = (np.zeros_like(t), np.zeros_like(t))
                
            # generate uniformly sampled u coefficients then calculate the state using numerical integration
            u = generate_controls(t, x, n_samples, n_t_coeffs, n_x_coeffs, u_max=u_max)
            y = torch.tensor( solve_state_eq(u.numpy(), y_IC, y_BCs, diffusion_coeff, t_span, x_span), dtype=torch.float32 )
            if add_noise:
                y += noise
            # return u and y as (n_samples, N_t*N_x) shaped arrays
            data["u"] = u#.flatten(start_dim=1, end_dim=2)
            data["tx"] = torch.cartesian_prod(t, x)[None].repeat(n_samples,1,1)
            data["y"] = y#.flatten(start_dim=1, end_dim=2)
            return data
        
        else:
            # generates adjoint p by sampling y uniformly
            y = generate_controls(x, basis, n_samples, coeff_range, n_coeffs)
            p = torch.tensor( solve_adjoint_eq(y.detach().numpy(), y_d=y_d.detach().numpy(), pf=boundary_condition), dtype=torch.float32 )
            if add_noise:
                p += noise
            data["y"] = y
            data["x"] = x.view(N,1).repeat(n_samples,1,1)
            data["p"] = p
            return data
    """
    else:
        # samples state uniformly and calculates the associated control
        dbernstein_basis = torch.tensor( [dbernstein_nk_dx(n_coeffs-1,k) for k in range(n_coeffs)], dtype=torch.float32)
        if not generateAdjoint:
            # then generate state data: y' = y + u
            y_coeffs = 2*coeff_range*torch.rand(size=(n_samples,n_coeffs)) - coeff_range
            y_coeffs[:,-1] = boundary_condition # so that boundary condition y(0) = y0 is met
            y = torch.einsum('ij,jk...->ik...', y_coeffs, bernstein_basis)
            dydx = torch.einsum('ij,jk...->ik...', y_coeffs, dbernstein_basis)
            u = dydx - y
            data[:,0,...] = u
            data[:,1,...] = y + noise
            return data
        
        else:
            # generate adjoint data: -p' = p + (y-y_d)
            p_coeffs = 2*coeff_range*torch.rand(size=(n_samples,n_coeffs)) - coeff_range
            p_coeffs[:,0] = boundary_condition # adjoint has terminal condition
            dpdx = torch.einsum('ij,jk...->ik...', p_coeffs, dbernstein_basis)
            y = - dpdx - p + y_d
            data[:,0,...] = y
            data[:,1,...] = p + noise
            return data"""

def normalise_tensor(t, dim):
    # normalises a tensor along a given dim by subtracting mean and dividing by (uncorrected) std.dev
    return (t - t.mean(dim=dim, keepdim=True)) / (t.std(dim=dim, keepdim=True) + 1e-6)