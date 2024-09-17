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

def grad_descent(f, grad_f, x0, max_no_iters=100, return_full_history=False, eps=1e-4, print_every=None):
    """
    Implements steepest descent: descent direction p = - gradient
    Arguments:
        f (function) : function to minimise
        grad_f (function) : gradient of f
        x0 (numpy array) : initial solution
        max_no_iters (int) : maximium number of iterations
        return_full_history (bool) : if True, a list of every point x_i in the optimisation trajectory is returned. If False, only the final x is returned.
        print_every (int or None) : the interval between printing of progress updates
    Returns:
        if return_full_history:
            x_history (list of np arrays) : list of the points x_i from every iteration i
        else:
            x (np array) : the final solution
        cost_history (list) : list of function values f(x_i) at every iteration i
        grad_history (list) : list of the norm of the gradient grad_f(x_i) at every iteration i
    """
    x = x0
    x_history = [x0]
    cost_history = []
    cost_history.append(f(x0))
    grad_history = []
    grad_history.append(np.linalg.norm(grad_f(x0)))
    i = 0
    
    while np.linalg.norm(grad_f(x))>=eps*grad_history[0]:
        if print_every:
            if i%print_every==0:
                print("{:g}/{:g}".format(i, max_no_iters), "Cost = {:.3e}".format(cost_history[i]))
        if i<max_no_iters:
            p = - grad_f(x) # steepest descent
            s = linesearch(f, grad_f, p, x)
            #s = 1/(i+1)
            x = x + s*p
            i+=1
            cost_history.append(f(x))
            grad_history.append(np.linalg.norm(grad_f(x)))
            if return_full_history:
                x_history.append(x)
        else:
            break
    
    if return_full_history:
        return x_history, cost_history, grad_history
    else:
        return x, cost_history, grad_history

def conjugate_gradient(f, grad_f, x0, max_no_iters=100, return_full_history=False, eps=1e-4, print_every=None):
    """
    Implements conjugate gradient: 
    Arguments:
        f (function) : function to minimise
        grad_f (function) : gradient of f
        x0 (numpy array) : initial solution
        max_no_iters (int) : maximium number of iterations
        return_full_history (bool) : if True, a list of every point x_i in the optimisation trajectory is returned. If False, only the final x is returned.
        print_every (int or None) : the interval between printing of progress updates
    Returns:
        if return_full_history:
            x_history (list of np arrays) : list of the points x_i from every iteration i
        else:
            x (np array) : the final solution
        cost_history (list) : list of function values f(x_i) at every iteration i
        grad_history (list) : list of the norm of the gradient grad_f(x_i) at every iteration i
    """
    
    x = x0
    x_history = [x0]
    cost_history = []
    cost_history.append(f(x0))
    grad_history = []
    grad_history.append(np.linalg.norm(grad_f(x0)))
    i = 0
    p = -grad_f(x) # initial descent direction
    r_old = p # residual r = 0 - grad(f)
    r_new = p
    while np.linalg.norm(grad_f(x))>eps*grad_history[0]:
        
        if print_every:
            if i%print_every==0:
                print("{:g}/{:g}".format(i,max_no_iters), "Cost = {:.3e}".format(cost_history[i]))
        if i<max_no_iters:
            s = linesearch(f, grad_f, p, x) # line search
            # update solution
            x = x + s*p
            # now update search direction p
            r_new = -grad_f(x)
            p = r_new + np.sum(r_new*r_new)/np.sum(r_old*r_old) * p
            r_old = r_new
            i+=1
            cost_history.append(f(x))
            grad_history.append(np.linalg.norm(grad_f(x)))
            if return_full_history:
                x_history.append(x)
        else:
            break
    if return_full_history:
        return x_history, cost_history, grad_history
    else:
        return x, cost_history, grad_history


def linesearch(f,grad_f,p,x,starting_stepsize=4.):
    stepsize = starting_stepsize
    alpha = 0.5e-4 # desired decrease
    
    while f(x + stepsize*p) > f(x) + alpha*stepsize*grad_f(x).ravel()@p.ravel():
        stepsize = 0.5*stepsize # half step size (backtracking)
        
        if stepsize < 1e-5: # no decrease found
            return stepsize
    
    return stepsize

