# -*- coding: utf-8 -*-
"""
Contains numerical optimization methods:
    - steepest descent
    - non-linear conjugate gradient
also a method for a simple Armijo line search (could be expanded to Wolffe search)
+ LU decomp method for the fun of it
(March 24)

(May 24): made it so descent routines return cost history + grad. norm history
"""
import numpy as np

def grad_descent(f, grad_f, x0, eps=1e-4, max_no_iters=10000):
    x = x0
    cost_history = []
    cost_history.append(f(x0))
    grad_history = []
    grad_history.append(np.linalg.norm(grad_f(x0)))
    i = 0
    while np.linalg.norm(grad_f(x))>eps:
        if i%(max_no_iters//50)==0:
            print(100*np.round(i/max_no_iters,2), "%")
        if i<max_no_iters:
            p = - grad_f(x) # steepest descent
            s = linesearch(f, grad_f, p, x)
            x = x + s*p
            i+=1
            cost_history.append(f(x))
            grad_history.append(np.linalg.norm(grad_f(x)))
        else:
            break
    
    return x, cost_history, grad_history

def conjugate_gradient(f, grad_f, x0, eps=1e-4, max_no_iters=10000):
    x = x0
    cost_history = []
    cost_history.append(f(x0))
    grad_history = []
    grad_history.append(np.linalg.norm(grad_f(x0)))
    i = 0
    p = -grad_f(x) # initial descent direction
    rOld = p # residual r = 0 - grad(f)
    rNew = p
    while np.linalg.norm(grad_f(x))>eps:
        if i%(max_no_iters//200)==0:
            print(200*np.round(i/max_no_iters,2), "%")
        if i<max_no_iters:
            s = linesearch(f, grad_f, p, x) # line search
            # update solution
            x = x + s*p
            # now update search direction p
            rNew = -grad_f(x)
            p = rNew + rNew.T@rNew/(rOld.T@rOld) * p
            rOld = rNew
            i+=1
            cost_history.append(f(x))
            grad_history.append(np.linalg.norm(grad_f(x)))
        else:
            break
    
    return x, cost_history, grad_history


def linesearch(f,grad_f,p,x,starting_stepsize=4.):
    stepsize = starting_stepsize
    alpha = 0.5e-4 # desired decrease
    while f(x + stepsize*p) > f(x) + alpha*stepsize*grad_f(x)@p:
        stepsize = 0.5*stepsize # half step size (backtracking)
        if stepsize < 1e-5: # no decrease found
            #print("shucks")
            return stepsize
    return stepsize


def LUsolve(A,b):
    N = len(A)
    # compute LU factorization
    L = np.eye(N)
    U = np.zeros_like(A)
    U[0,:] = A[0,:]
    L[:,0] = A[:,0]/U[0,0]
    for i in range(1,N):
        U[i,:] = A[i,:] - L[i,:i]@U[:i,:]
        L[:,i] = (A[:,i] - L[:,:i]@U[:i,i])/U[i,i]
    
    # solve Lz = b, forward substitution
    z = np.zeros(N)
    z[0] = b[0]
    for i in range(1,N):
        z[i] = b[i] - L[i,:]@z
    
    # solve Ux = z, back substitution
    x = np.zeros(N)
    x[-1] = z[-1]/U[-1,-1]
    for i in range(N-2,-1,-1):
        x[i] = (z[i] - U[i,:]@x)/U[i,i]
    
    return x