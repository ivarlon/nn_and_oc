# -*- coding: utf-8 -*-
"""
Generate data (u,y) for the Poisson equation
"""
import numpy as np
from solve_poisson import *
import torch

def generate_controls(x1, x2,
                      n_samples,
                      max_n_combs=10,
                      min_n_combs=1,
                      normalize=True,
                      u_max=50.):
    # generate controls u(x1,x2) as a superposition of Gaussians
    L = x1[-1] # length of grid dimensions
    N = len(x1)
    
    U = []
    
    # generate a tensor that describes how many gaussians to combine for each generated control
    number_of_gaussian_combs = torch.randint(low=min_n_combs, high=max_n_combs+1, size=(n_samples,))
    # count the number of controls per number of gaussian combinations
    number_of_controls_per_number_of_combs = torch.stack([number_of_gaussian_combs==i for i in range(min_n_combs, max_n_combs+1)], axis=0).sum(axis=1)
    
    R = torch.stack(torch.meshgrid(x1, x2, indexing="ij"), 
                    axis=-1)
    
    for i in range(max_n_combs - min_n_combs +1):
        m = min_n_combs + i
        if number_of_controls_per_number_of_combs[i] == 0:
            # skip if there is no gaussian mixture for this number
            continue
        coeffs = 2.*torch.rand(size=(number_of_controls_per_number_of_combs[i], m)) - 1.# weight for each Gaussian in mixture
        coeffs /= coeffs.sum(dim=1, keepdim=True) # normalize sum of coeffs to be equal to 1
        coeffs*= 0.99*u_max*torch.rand(size=(number_of_controls_per_number_of_combs[i],1)) + 0.01*u_max # scale coefficients again
        locations = L*torch.rand(size=(number_of_controls_per_number_of_combs[i], m, 2)) # location of each gaussian
        
        displacements = R[None,None]-locations[:,:,None,None]
        
        spreads = L*(0.1*torch.rand(size=(number_of_controls_per_number_of_combs[i], m)) + 1e-1) # spread of each gaussian in directions of eigenvectors of cov matrix
        
        square_inside_exponential = torch.sum(displacements**2, axis=-1)/spreads[:,:,None,None]**2
        
        # scale gaussians by inverse spread squared
        gaussian_mix = torch.sum( torch.exp(-0.5*square_inside_exponential)*(coeffs/spreads**2)[:,:,None,None], axis=1)/(2*torch.pi)
        
        if normalize:
            # scale samples that exceed a threshold u_max
            samples_exceeding_max = torch.unique(torch.where(gaussian_mix.abs()>u_max)[0])
            
            if len(samples_exceeding_max) > 0:
                gaussian_mix[samples_exceeding_max] /= gaussian_mix[samples_exceeding_max].abs().max(dim=1, keepdim=True)[0].max(dim=2,keepdim=True)[0]/u_max
        
        U.append(gaussian_mix)
    
    U = torch.cat(U)
    
    return U

def generate_data(N,
                  L,
                  BCs=None,
                  n_samples=1024,
                  u_max=100.,
                  generate_adjoint=False,
                  y_d=None, 
                  seed=None,
                  add_noise=False):
    """
    
    """
    # set up data dict
    data = {}
    
    # domain
    
    x1 = torch.linspace(0., L, N)
    x2 = torch.linspace(0., L, N)
    
    # seed RNG
    if seed:
        torch.manual_seed(seed)
    
    if add_noise:
        noise = 1e-2*u_max*torch.randn(size=(n_samples, N, N, 1), dtype=torch.float32)
    
    if BCs is None:
        BCs = [np.zeros(N) for i in range(4)]
    
    if not generate_adjoint:
        # generate uniformly sampled u coefficients then calculate the state using numerical integration
        u = generate_controls(x1, x2, n_samples, min_n_combs=5, max_n_combs=15, normalize=True, u_max=u_max)
        y = torch.tensor( solve_poisson(u.numpy(), BCs), dtype=torch.float32 )
        if add_noise:
            y += noise
        # return u and y as (n_samples, N, N) shaped arrays
        data["u"] = u
        data["y"] = y
    
    else:
        # generates adjoint p by sampling y uniformly and solving adjoint eq for p
        y = generate_controls(x1, x2, n_samples, normalize=True, u_max=u_max)
        p = torch.tensor( solve_poisson((y-y_d).numpy(), BCs), dtype=torch.float32 )
        if add_noise:
            p += noise
        data["y-y_d"] = (y-y_d)
        data["p"] = p
    
    data["r"] = torch.cartesian_prod(x1, x2)[None].expand(n_samples, N**2, 2)    
    
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
    R = data["r"]
    
    assert n_samples > n_combinations, "number of samples must be greater than number of combinations"
    
    # index array for which samples to combine
    idx = torch.stack([ torch.randperm(n_samples)[:n_combinations] for i in range(n_augmented_samples) ])
    
    # sample lin.comb. coefficients from hypercube [-maxcoeff,maxcoeff]^n_comb
    coeffs = 2*max_coeff*torch.rand(size=(n_augmented_samples,n_combinations)) - max_coeff
    
    # normalise sum of each coeff vector to 1 (to ensure lin. comb. respects boundary conditions) (assume coeffs don't sum to zero, but unlikely)
    coeffs = coeffs/(coeffs.sum(dim=1, keepdims=True))
    if not adjoint:
        y_aug = torch.einsum('nc..., nc...->n...', coeffs, y[idx] )
        
        # make sure variance of new samples is similar to original samples. Just do a convex combination
        max_y = torch.max(torch.var(y, dim=tuple(torch.arange(1,y.ndim)))) # find max std. dev. of input samples. augmented samples should not greatly exceed this
        unacceptably_large_samples = torch.where( torch.var( y_aug, dim=tuple(torch.arange(1,y.ndim))) > max_y)
        n_bad_samples = len(unacceptably_large_samples)
        convex_coeffs = torch.rand(size=(n_bad_samples, n_combinations))
        convex_coeffs = convex_coeffs/(convex_coeffs.sum(dim=1, keepdims=True))
        new_idx = torch.stack([ torch.randperm(n_samples)[:n_combinations] for i in range(n_bad_samples) ])
        y_aug[unacceptably_large_samples] = torch.einsum('nc..., nc...->n...', convex_coeffs, y[new_idx] )
        
        u_aug = torch.einsum('nc..., nc...->n...', coeffs, u[idx] )
        u_aug[unacceptably_large_samples] = torch.einsum('nc..., nc...->n...', convex_coeffs, u[new_idx] )
        
        # make sure controls don't exceed max threshold
        u_max = torch.max(u[idx].abs())
        samples_exceeding_max = torch.unique(torch.where(u_aug.abs()>u_max)[0])
        n_samples_above_threshold = len(samples_exceeding_max)
        
        if n_samples_above_threshold>0:
            convex_coeffs = torch.rand(size=(n_samples_above_threshold, n_combinations))
            convex_coeffs = convex_coeffs/(convex_coeffs.sum(dim=1, keepdims=True))
            new_idx = torch.stack([ torch.randperm(n_samples)[:n_combinations] for i in range(n_samples_above_threshold) ])
            y_aug[samples_exceeding_max] = torch.einsum('nc..., nc...->n...', convex_coeffs, y[new_idx] )
            u_aug[samples_exceeding_max] = torch.einsum('nc..., nc...->n...', convex_coeffs, u[new_idx] )
        
        data["u"] = torch.cat((u, u_aug))
        data["y"] = torch.cat((y, y_aug))
    else:
        p_aug = torch.einsum('nc..., nc...->n...', coeffs, p[idx] )
        
        # make sure variance of new samples is similar to original samples. Just do a convex combination
        max_p = torch.max(torch.var(p, dim=tuple(torch.arange(1,p.ndim)))) # find max std. dev. of input samples. augmented samples should not greatly exceed this
        unacceptably_large_samples = torch.where( torch.var( p_aug, dim=tuple(torch.arange(1,p.ndim))) > max_p)
        n_bad_samples = len(unacceptably_large_samples)
        convex_coeffs = torch.rand(size=(n_bad_samples, n_combinations))
        convex_coeffs = convex_coeffs/(convex_coeffs.sum(dim=1, keepdims=True))
        new_idx = torch.stack([ torch.randperm(n_samples)[:n_combinations] for i in range(n_bad_samples) ])
        p_aug[unacceptably_large_samples] = torch.einsum('nc..., nc...->n...', convex_coeffs, p[new_idx] )
        
        y_y_d_aug = torch.einsum('nc..., nc...->n...', coeffs, y_y_d[idx] )
        y_y_d_aug[unacceptably_large_samples] = torch.einsum('nc..., nc...->n...', convex_coeffs, y_y_d[new_idx] )
        
        data["y-y_d"] = torch.cat((y_y_d, y_y_d_aug))
        data["p"] = torch.cat((p, p_aug))
    data["r"] = torch.cat((R, R[0].expand(n_augmented_samples,R.shape[1],R.shape[2])))