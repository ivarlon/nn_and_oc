# -*- coding: utf-8 -*-
"""
Visualising NN training results

Reads saved npy arrays containing loss histories
"""

# open data files: train loss histories, test loss, training time
# visualise: train loss for each hyperparameter tuple - title hyperparameters, label train_time
#            test loss: scatterplot - hyperparameter tuple #i on x-axis, test loss on y-axis
#            training time: labels on train loss

import numpy as np
import matplotlib.pyplot as plt
import glob

train_loss_files = glob.glob(r".\data\FNO_loss_train*.npy")
n_train_files = len(train_loss_files)
test_loss_files = glob.glob(r".\data\FNO_loss_test*.npy")

weight_penalties = [str(0.0), str(1e-3), str(1e-2)]
n_weight_penalties = len(weight_penalties)
n_models = 5

figs_and_axs = [plt.subplots(ncols=3, nrows=3, figsize=(12,12)) for i in range(n_weight_penalties)]

for i, filename in enumerate(train_loss_files):
    train_loss = np.load(filename)
    weight_penalty = filename.split("_")[3]
    n_layers = filename.split("_")[4]
    d_v = filename.split("_")[5]
    n_model = filename.split("_")[6]
    for j in range(len(weight_penalties)):
        if weight_penalty == weight_penalties[j]:
            axs = figs_and_axs[j][1]
    row_idx = i%(n_train_files//n_weight_penalties)//n_models//3
    col_idx = i%(n_train_files//n_weight_penalties)//n_models%3
    axs[row_idx, col_idx].plot(range(len(train_loss)), train_loss)
    axs[row_idx, col_idx].set_yscale("log")
    axs[row_idx, col_idx].set_title(n_layers + "," + d_v + "," + weight_penalty)
    axs[row_idx, col_idx].set_ylim([1e-4, 1e0])
    if i%n_models==0:
        test_loss = np.around( 1e4*np.mean( np.load(test_loss_files[i//n_models]) ), 1)
        axs[row_idx, col_idx].text(x=0., 
                                   y=3e-4,
                                   s='Test error =' + str(test_loss) + "e-4")
plt.show()
for i, (fig,_) in enumerate(figs_and_axs):
    fig.savefig("train_loss_{}.pdf".format(i))
"""
test_loss_files = glob.glob(r".\data\FNO_loss_test*.npy")

fig, ax = plt.subplots(figsize=(15,6))
xticks = []
xticklabels = []

for i, filename in enumerate(test_loss_files):
    test_loss = np.load(filename)
    weight_penalty = filename.split("_")[3]
    n_layers = filename.split("_")[4]
    d_v = filename.split("_")[5].split(".")[0]
    ax.scatter(i*np.ones(4), test_loss)
    xticks.append(i)
    xticklabels.append("(" + n_layers + "," + d_v + ")")
    
    print("mean test error for model", weight_penalty, n_layers, d_v, ":", np.around(1e4*np.mean(test_loss),2), "e-4")
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
fig.savefig("test_loss.pdf")
plt.show()"""