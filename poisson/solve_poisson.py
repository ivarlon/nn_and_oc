# -*- coding: utf-8 -*-
"""
Numerical solver for the Poisson equation.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron, eye, csr_matrix
import time
def solve_poisson(u, BCs, L=1.):
    """
    Solves the Poisson eq. with source u
    u : numpy array shape (N_samples, N, N)
    BCs : list or tuple with boundary conditions: ( y(x1,0), y(1,x2), y(x1,1), y(0,x2) )
    """
    n_samples = u.shape[0]
    N = u.shape[1]
    
    # flatten matrix to vector, discarding values in corners of grid
    u_flatten = u[:,1:-1,1:-1].transpose(0,2,1).reshape(n_samples, (N-2)**2) # numpy stacks rows when reshaping, so tranpose rows<->columns to get columns stacked
    
    # generate boundary condition vector
    unit_vector1, unit_vectorN_2 = np.eye(N-2)[[0,-1]] # create unit vectors
    
    boundary_conditions = kron(unit_vector1, BCs[0][1:-1]) + kron(unit_vectorN_2, BCs[2][1:-1]) \
        + kron(BCs[1][1:-1], unit_vectorN_2) + kron(BCs[3][1:-1], unit_vector1)
    
    h = L/(N-1) # step size
    
    def generate_A(N):
        M = N-2
        B = -4*np.eye(M) + np.diag(np.ones(M-1), 1) + np.diag(np.ones(M-1), -1)
        A = kron(eye(M), B, format="csr") + kron(eye(M, k=1), np.eye(M), format="csr") + kron(eye(M, k=-1), np.eye(M), format="csr")
        return A
    
    A = generate_A(N)
    b = -h**2 * u_flatten - boundary_conditions.toarray().repeat(n_samples, axis=0)
    
    y_interior = spsolve(A, b.transpose()).transpose()
    
    y = np.zeros(shape=(n_samples, N, N))
    y[:,1:-1,1:-1] = y_interior.reshape(n_samples, N-2, N-2).transpose(0,2,1) # do inverse of reshape and transpose done to u above
    y[:,:,0] = BCs[0]
    y[:,:,-1] = BCs[2]
    y[:,-1,:] = BCs[1]
    y[:,0,:] = BCs[-1][-1]
    
    return y

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 32
    
    x1, x2 = np.meshgrid(*(np.linspace(0.,1.,N) for i in range(2)), indexing='ij')
    
    BCs = [np.zeros(N) for i in range(4)]
    
    """
    #point sources
    u = np.zeros(shape=(1, N, N))
    for i in range(2):
        for j in range(2):
            u[:,(2*i+1)*N//4, (2*j+1)*N//4] = 100000.
    """
    #BCs[0] = np.sin(2*np.pi*np.linspace(0.,1.,N))**2*(np.exp(1.)-1.)
    u = - (np.exp(-(x2-1.))*np.sin(2*np.pi*x1)**2  + 8*np.pi**2*(1. - 2*np.sin(2*np.pi*x1)**2)*(np.exp(-(x2-1.)) - 1.) )
    u = u[None]
    sigma = .1
    x0, y0 = 0.8, 0.99
    exp_arg = (x1-x0)**2 + (x2-y0)**2
    u = 5000./(2*np.pi*sigma**2)*np.exp(-0.5*exp_arg/sigma**2)
    x0, y0 = 0.3, 0.99
    exp_arg = (x1-x0)**2 + (x2-y0)**2
    u += 5000./(2*np.pi*sigma**2)*np.exp(-0.5*exp_arg/sigma**2)
    x0, y0 = 0.01, 0.95
    exp_arg = (x1-x0)**2 + (x2-y0)**2
    u += 5000./(2*np.pi*sigma**2)*np.exp(-0.5*exp_arg/sigma**2)
    u/=3.
    """sigma = 1.8
    x0, y0 = 0.9, 0.01
    exp_arg = (x1-x0)**2 + (x2-y0)**2
    u += 1/(2*np.pi*sigma**2)*np.exp(-0.5*exp_arg/sigma**2)
    sigma = 1.4
    x0, y0 = 0.1, 0.01
    exp_arg = (x1-x0)**2 + (x2-y0)**2
    u += 1/(2*np.pi*sigma**2)*np.exp(-0.5*exp_arg/sigma**2)
    u0 = 100."""
    #u += u0
    u = u[None]
    
    
    #y_analytic = np.sin(2*np.pi*x1)**2 * (np.exp(-(x2-1.)) - 1.)
    
    y = solve_poisson(u, BCs)
    fig = plt.contourf(x1, x2, y[0], levels=np.linspace(y.min(),y.max()))
    cbar = plt.colorbar()
    plt.show()
    fig = plt.contourf(x1, x2, u[0], levels=np.linspace(u.min(),u.max()))
    cbar = plt.colorbar()
    plt.show()
    #fig2 = plt.contourf(x1, x2, y_analytic, levels=np.linspace(y_analytic.min(),y_analytic.max()))
    #cbar = plt.colorbar()
    #plt.show()
    
    def test_PDE(y,u):
        h = 1./(N-1)
        y_x = (y[1:] - y[:-1])/h
        y_xx = (y_x[1:] - y_x[:-1])/h
        y_y = (y[:,1:] - y[:,:-1])/h
        y_yy = (y_y[:,1:] - y_y[:,:-1])/h
        
        res = y_xx[:,1:-1] + y_yy[1:-1] + u[1:-1,1:-1]
        return res.mean(), res.std()
        
    print(test_PDE(y[0],u[0]))