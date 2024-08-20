    # -*- coding: utf-8 -*-
"""
Implements the Crank-Nicolson scheme to solve the heat equation.
"""

import numpy as np

from scipy.linalg import solveh_banded

def crank_nicolson(y_IC, y_BCs, D, u, t0=0., tf=1., x0=0.,xf=1.):
        
        # y_IC = y(t=0, x)
        # y_BCs = ( y(t, x=0), y(t, x=L) )
        # D is diffusion coefficient
        # 
        # This can also solve the adjoint heat eq -p_t = Dp_xx + J_y
        # provided that u:= J_y and the output is flipped along the time axis
        # This is because A <-> B when dt becomes -dt.
        n_samples = u.shape[0]
        N_t = u.shape[1]
        N_x = len(y_IC)
        dx = (xf-x0)/(N_x-1)
        dt = (tf-t0)/(N_t-1)
        Ddx2 = D/dx**2
        r = dt*Ddx2
        # set up tridiagonal matrices
        M = N_x - 2 # no. of unknowns at locations xj j=1,...,Nx-2 at time ti, with solution known at x[0],x[-1]
        A_diags = np.stack( [ (1. + r)*np.ones(M), np.concatenate([-0.5*r*np.ones(M-1), np.zeros(1)]) ]) # main diag and sub/superdiags of A
        B = (1. - r)*np.eye(M) + np.diag(0.5*r*np.ones(M-1), 1) + np.diag(0.5*r*np.ones(M-1), -1)
        A = (1. + r)*np.eye(M) - np.diag(0.5*r*np.ones(M-1), 1) - np.diag(0.5*r*np.ones(M-1), -1)
        #A = np.diag(A_diags[0]) + np.diag(A_diags[1,:-1], 1) - np.diag(A_diags[1,:-1], -1)
        # set up solution array
        y = np.zeros(shape=(n_samples, N_t, N_x))
        y[:,0,:] = y_IC
        y[:,:,0] = y_BCs[0]; y[:,:,-1] = y_BCs[1]
        
        # implement BCs: y(0,t)=f(t), y(L,t)=g(t)
        BC_arr = np.zeros(shape=(N_t, M))
        BC_arr[:,0] = y_BCs[0]
        BC_arr[:,-1] = y_BCs[1]
        for i in range(N_t-1):
            #if i%200==0:
            #    print(np.round(i/N_t*100,3))
            # solve Ay[i+1,interior] = b
            # b:= By[i,interior] + dt/2*(u[i+1]+u[i]) + r(y[i,boundary] + y[i+1,boundary])
            b = np.einsum('kx, bx->bk', B, y[:,i,1:-1]) + 0.5*dt*(u[:,i+1,1:-1] + u[:,i,1:-1]) + 0.5*r*(BC_arr[i] + BC_arr[i+1])
            #print(A_diags.shape, b.T.shape)
            #eigvals, eigvecs = np.linalg.eigh(A)
            #print(eigvals.min())
            y[:,i+1,1:-1] = solveh_banded(A_diags, b.T, lower=True).T
            #y[:,i+1,1:-1] = np.linalg.solve(A, b.T).T
        return y
    
if __name__ == "__main__":
        import matplotlib.pyplot as plt
        import time
        np.random.seed(10)
        D = 1e-1 # unrealistically large diffusion coeff
        N_t = 64
        N_x = 64
        L = 1.
        x0 = 0.; xf = x0 + L
        T = 1.
        t0 = 0.; tf = t0 + T
        x = np.linspace(x0,xf,N_x)
        t = np.linspace(t0,tf,N_t)
        x_grid,t_grid = np.meshgrid(x,t)
        delta_x = x[1]-x[0]
        delta_t = t[1]-t[0]
        def test_PDE(p, y, y_d):
            N_t = p.shape[0]
            N_x = p.shape[1]
            delta_t = T/(N_t-1)
            delta_x = L/(N_x-1)
            p_t = (p[2:]-p[:-2])/(2*delta_t)
            p_x = (p[:,2:]-p[:,:-2])/(2*delta_x)
            p_xx = (p_x[:,1:]-p_x[:,:-1])/delta_x
            r = -0.5*(p_t[:,2:-1] + p_t[:,1:-2]) + D*p_xx[1:-1] - 0.5*( (y - y_d)[1:-1,1:-2] + (y - y_d)[1:-1,2:-1])
            #print(r.max(), r.min(), np.abs(r).mean())
            print(r.mean(), r.var())
        """y0 = np.sin(4*np.pi*x)
        
        X,T = np.meshgrid(x,t)
        u = 10*np.exp(X)[None,:,:]
        n_samples = 1000
        n_t_coeffs = 6
        n_x_coeffs = 6
        t_coeffs = np.random.randint(low=1, high=4, size=(n_samples, n_t_coeffs))
        t_pols = np.stack([np.polynomial.Legendre(coef=t_coef, domain=(t0,tf))(T) for t_coef in t_coeffs])
        x_coeffs = np.random.randint(low=-2, high=4, size=(n_samples, n_x_coeffs))
        x_pols = np.stack([np.polynomial.Legendre(coef=x_coef, domain=(x0,xf))(X) for x_coef in x_coeffs])
        #u = np.einsum('btx, btx-> btx', t_pols, x_pols)
        start = time.time()
        y = crank_nicolson(y0, (0*t, 0*t), D, u)
        end = time.time()
        print("took {} s".format(end-start))
        sample_idx = 0
        time_idx = [0, N_t//3, N_t//2, N_t-1]
        for t in time_idx:
            plt.plot(x, y[sample_idx, t], label=str(tf*t/N_t))
        plt.legend()"""
        
        # compare to anal. sol. of heat eq backwards in time
        k = np.pi/L
        pf = np.zeros_like(x)*T**2 #np.sin(k*x)
        p_BC = (np.sin(3*np.pi/T *np.flip(t)), np.sin(3*np.pi/T *np.flip(t)))
        time_start = time.time()
        y = -3*np.pi/T*np.cos(3*np.pi/T *np.flip(t_grid))
        p = crank_nicolson(pf, p_BC, D, t0=t0, tf=tf, x0=x0, xf=xf, u=y[None])[0]
        p = np.flip(p, axis=0)
        print("Time", time.time() - time_start)
        plt.contourf(t_grid,x_grid,p); plt.colorbar()
        plt.show()
        p_anal = np.sin(3*np.pi/T*t_grid)#x_grid**2 #np.exp(D*k**2*(t_grid-T))*np.sin(k*x_grid)
        plt.contourf(t_grid, x_grid, p_anal); plt.colorbar()
        plt.show()
        plt.contourf(t_grid, x_grid, p - p_anal); plt.colorbar()
        plt.show()
        test_PDE(p, np.flip(y, axis=0), np.zeros_like(p))
        test_PDE(p_anal, np.flip(y,axis=0), np.zeros_like(p))