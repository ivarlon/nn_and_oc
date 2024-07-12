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
    dim_y = len(y0) # dimension of y
    y = np.zeros(shape=(n_batches, n_points, dim_y))
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

def solveStateEq(u, y0=1., batched=True):
    # if batched=False, u has shape (N,...)
    # solve ODE y' = -y + u
    state_eq = lambda x, y: -y
    if not batched:
        u = u[None,:]
    N = u.shape[1]
    y = improved_forward_euler(state_eq, [y0], n_points=N, x_span=(0.,1.), u=u)
    if not batched:
        return y[0]
    else:
        return y
    
    
def solveAdjointEq(y, y_d, pf=0., batched=True):
    # if batched=False, y, y_d have shape (N,...)
    # solve ODE -p' = -p + (y-y_d)
    # backwards in time: p(t-1) = p(t) - dt*p' = p(t) + dt (-p')
    adjoint_eq = lambda x, p: -p
    if not batched:
        y = y[None,:]
        y_d = y_d[None,:]
    N = y.shape[1]
    p = improved_forward_euler(adjoint_eq, [pf], n_points=N, x_span=(1.,0.), u=y-y_d)
    p = np.flip(p, axis=1)
    if not batched:
        return p[0]
    else:
        return p

def normalised_bernstein(coeffs, x):
    # returns Bernstein basis for P_{n_coeffs-1}
    
    n_coeffs = coeffs.shape[-1]
    n = n_coeffs - 1
    
    # define Bernstein polynomials
    binom_coeff = lambda n, k: np.prod([(n-i)/(k-i) for i in range(k)] ) 
    bernstein_nk = lambda n, k: binom_coeff(n,k) * x**k * (1.-x)**(n-k)
    
    # define matrix of normalisation factors: normalised b_i is a lin. comb. of non-normalised B_k
    normalisation_factors = torch.stack([ np.sqrt(2*(n-i)+1)*
            torch.tensor([ (-1)**k * binom_coeff(2*n+1-k, i-k)*binom_coeff(i,k)/binom_coeff(n-k, i-k) for k in range(i+1)] + [0. for k in range(i+1,n_coeffs)], dtype=torch.float32)
            for i in range(n_coeffs)])
    # define corresponding matrix of non-orthonormal b.stein basis pol.s
    non_normalised_bstein = torch.stack([
            torch.stack([ bernstein_nk(n-k, i-k) for k in range(i+1)] + [torch.zeros_like(x) for k in range(i+1,n_coeffs)]) 
            for i in range(n_coeffs)])
    
    # define orthonormal b.stein basis
    normalised_bstein = torch.einsum("ij...,ij...->i...", normalisation_factors, non_normalised_bstein)
    
    return torch.einsum("c...,c...->...", coeffs, normalised_bstein)

def bernstein(coeffs):
    # returns Bernstein basis for P_{n_coeffs-1}
    
    n_coeffs = coeffs.shape[-1]
    n = n_coeffs - 1
    
    # define Bernstein polynomials
    binom_coeff = lambda n, k: np.prod([(n-i)/(k-i) for i in range(k)] ) 
    bernstein_nk = lambda n, k, x: binom_coeff(n,k) * x**k * (1.-x)**(n-k)
    
    bstein = torch.stack([
            torch.stack([ bernstein_nk(n-k, i-k) for k in range(i+1)] + [torch.zeros_like(x) for k in range(i+1,n_coeffs)]) 
            for i in range(n_coeffs)])

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
    elif basis=="Bernstein":
        polynomial = lambda coeffs, x: bernstein(coeffs, x)
    else:
        print("Enter a valid polynomial basis")
        return None
    u_coeffs = 2*coeff_range*torch.rand(size=(n_samples,n_coeffs)) - coeff_range
    u = torch.stack( [polynomial(coeffs, x) for coeffs in u_coeffs] )
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
    x = torch.linspace(0.,1.,N).view(N,-1)
    
    # seed RNG
    if seed:
        torch.manual_seed(seed)
    
    if add_noise:
        noise = 1e-2*coeff_range*torch.randn(size=(n_samples, N, 1), dtype=torch.float32)
    
    if sample_input_function_uniformly:
        
        if not generate_adjoint:
            # generate uniformly sampled u coefficients then calculate the state using numerical integration
            u = generate_controls(x, basis, n_samples, coeff_range, n_coeffs)
            y = torch.tensor( solveStateEq(u.detach().numpy(),boundary_condition), dtype=torch.float32 )
            if add_noise:
                noise = 1e-2*coeff_range*torch.randn(size=(n_samples, N, 1), dtype=torch.float32)
                y += noise
            data["u"] = u
            data["x"] = x.view(N,1).repeat(n_samples,1,1)
            data["y"] = y
            return data
        
        else:
            # generates adjoint p by sampling y uniformly
            y = generate_controls(x, basis, n_samples, coeff_range, n_coeffs)
            p = torch.tensor( solveAdjointEq(y.detach().numpy(), y_d=y_d.detach().numpy(), pf=boundary_condition), dtype=torch.float32 )
            if add_noise:
                noise = 1e-2*coeff_range*torch.randn(size=(n_samples, N, 1), dtype=torch.float32)
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