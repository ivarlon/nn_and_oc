# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:20:17 2024

Generates data (u_i, y_i) for exponential decay problem y' = -y + u
"""

import numpy as np
import torch

def solveStateEq(u, y0=1.):
    # y' = -y + u
    N = u.shape[1]
    y = np.zeros_like(u)
    y[:,0] = y0
    h = 1./(N-1)
    """D = np.roll(np.eye(N), 1, axis=1) + (-1)*np.roll(np.eye(N), 1, axis=0)
    D[0,0] = -1.; D[0,-1] = 0.; D[-1,0] = 0.; D[-1,-1] = 1.
    D = 1/h*D - np.eye(N)
    # know y0, so rearrange equation to get unknown [y1,...yN] on left side
    ... """
    # solve by improved forward Euler
    for i in range(N-2):
        dy_i1 = h*(-y[:,i] + u[:,i]) # gradient at yi
        dy_i2 = h*(-(y[:,i] + dy_i1) + u[:,i+1]) # approximated gradient at y(x(i+1))
        y[:,i+1] = y[:,i] + 0.5*(dy_i1 + dy_i2)
    y[:,-1] = y[:,-2] + h*(-y[:,-2] + 0.5*(u[:,-2]+u[:,-1]))
    
    return y
    
    
def solveAdjointEq(y, y_d, pN=0.):
    # -p' = -p + (y-y_d)
    N = y.shape[1]
    p = np.zeros_like(y)
    p[:,-1] = pN
    h = 1./(N-1)
    # improved forward Euler (backwards in time)
    for i in range(1,N-1):
        dp_i1 = h*(p[:,N-i] - (y[:,N-i] - y_d[N-i]) )
        dp_i2 = h*(p[:,N-i] - dp_i1 - (y[:,N-i-1] - y_d[N-i-1]) )
        p[:,N-1-i] = p[:,N-i] - 0.5*(dp_i1 + dp_i2)
    p[:,0] = p[:,1] - h*(p[:,1] - 0.5*(y[:,1] - y_d[1] + y[:,0] - y_d[0]))
    
    return p

def bernstein(coeffs, x):
    # returns Bernstein basis for P_{n_coeffs-1}
    
    n_coeffs = coeffs.shape[-1]
    n = n_coeffs - 1
    
    # define Bernstein polynomials
    binom_coeff = lambda n, k: np.prod([(n-i)/(k-i) for i in range(k)] ) 
    bernstein_nk = lambda n, k: binom_coeff(n,k) * x**k * (1.-x)**(n-k)
    #dbernstein_nk_dx = lambda n, k: n*( bernstein_nk(n-1,k) - bernstein_nk(n-1,k-1)) if k>=1 and k<n\
    #    else (n*bernstein_nk(n-1,k) if k<n else -n*bernstein_nk(n-1,k-1))
    #bernstein_basis = torch.stack([bernstein_nk(n_coeffs-1,k) for k in range(n_coeffs)])
    
    # define matrix of normalisation factors: normalised b_i is a lin. comb. of non-normalised B_k
    normalisation_factors = torch.stack([ np.sqrt(2*(n-i)+1)*
            torch.tensor([ (-1)**k * binom_coeff(2*n+1-k, i-k)*binom_coeff(i,k)/binom_coeff(n-k, i-k) for k in range(i+1)] + [0. for k in range(i+1,n_coeffs)], dtype=torch.float32)
            for i in range(n_coeffs)])
    # define corresponding matrix of non-orthonormal b.stein basis pol.s
    non_normalised_bernstein = torch.stack([
            torch.stack([ bernstein_nk(n-k, i-k) for k in range(i+1)] + [torch.zeros_like(x) for k in range(i+1,n_coeffs)]) 
            for i in range(n_coeffs)])
    
    # define orthonormal b.stein basis
    normalised_bernstein = torch.einsum("ij...,ij...->i...", normalisation_factors, non_normalised_bernstein)
    
    return torch.einsum("c...,c...->...", coeffs, normalised_bernstein)

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
                  sampleInputFunctionUniformly=True,
                  generateAdjoint=False,
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
    
    if sampleInputFunctionUniformly:
        
        if not generateAdjoint:
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
            p = torch.tensor( solveAdjointEq(y.detach().numpy(), y_d=y_d.detach().numpy(), pN=boundary_condition), dtype=torch.float32 )
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