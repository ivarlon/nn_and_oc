# -*- coding: utf-8 -*-
"""
Created on Fr
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from generate_data import *

N = 64

filenames = glob.glob('.\data2\FNO*.pt')
n_models = len(filenames)

test_data = generate_data(N, "Bernstein", n_samples=200, seed=420)
u_test = test_data[:,0]; y_test = test_data[:,1]

preds_for_each_model = []

for filename in filenames:
    model = torch.load(filename)
    preds = model(u_test[:,:,None])[...,0]
    preds_for_each_model.append(preds)

preds_for_each_model = torch.stack(preds_for_each_model)

B = 1000

def bootstrap(targets, preds, n, B=1000):
    n_samples = targets.shape[0]
    n_models = preds.shape[0]
    sample_indices = torch.randint(high=n_models, size=(B, n))
    bootstrap_preds = preds[sample_indices]
    mean_preds = torch.mean(bootstrap_preds, axis=1) # do mean over models for every bootstrap sample
    MSE = torch.mean( (targets - mean_preds)**2, axis=-1) # calculate MSE for every boostrap sample and every test sample
    
    m = torch.mean(MSE).item() # calculate mean of test MSEs (for each bootstrap) and mean of every bootstrap
    s = torch.std(torch.mean(MSE,axis=1), correction=1).item() # calculate sd of bootstrapped mean test MSEs (have to do mean over every test sample first)
    return m, s

means = []
sds = []
import time

t0 = time.time()
for n in range(1,n_models):
    print(n)
    m, s = bootstrap(y_test, preds_for_each_model, n)
    means.append(m)
    sds.append(s)
print("Took", round(time.time()-t0,1), "secs")