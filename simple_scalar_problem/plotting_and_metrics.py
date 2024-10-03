# -*- coding: utf-8 -*-
"""
Printing model metrics and plotting training history and predictions
"""
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import torch
import FNO
import DeepONet

model_name = "DON" # "DON"
problem = "state" # "adjoint"
#filename = glob.glob(".\loss*")[0]

model_list_filenames = glob.glob('.\{}_experiments_{}_15_models\models_list*.pkl'.format(problem, model_name))
metrics_filenames = glob.glob('.\{}_experiments_{}_15_models\metrics_*0.001.pkl'.format(problem, model_name))
loss_history_filenames = glob.glob('.\{}_experiments_{}_15_models\loss_history_*0.001.pkl'.format(problem, model_name)) #+ glob.glob('.\{}_experiments_{}\loss_history_2_[2*.001.pkl'.format(problem, model_name))

with open(model_list_filenames[0], 'rb') as infile:
    model_list = pickle.load(infile)

fig, axs = plt.subplots(2,4, figsize=(12,4.5))

for n, (lh_filename, metrics_filename) in enumerate(zip(loss_history_filenames, metrics_filenames)):
    if model_name=="DON":
        fig, axs = plt.subplots(1,3, figsize=(12,5))
        params = lh_filename.split("_")[-4:-1]
        fig.suptitle(params[0] + "," + params[1] + "," + params[2])
        with open(lh_filename, 'rb') as loss_file:
            lhs = pickle.load(loss_file)
            with open(metrics_filename, 'rb') as metrics_file:
                metrics = pickle.load(metrics_file)
                print(params)
                for i, (lh_d, lh_p, lh, lh_v) in enumerate(zip(lhs['data'], lhs['physics'], lhs['total'], lhs['validation'])):
                    if i>=8:
                        break
                    r2 = metrics["R2"][i].item()
                    axs[0].plot(np.arange(len(lh)), lh, linewidth=0.5, label="{:.2f}".format(r2)); axs[0].set_yscale("log"); axs[0].set_ylim([1e-5, 1e-1]); axs[0].set_title("Total loss")
                    axs[1].plot(np.arange(len(lh_d)), lh_d, np.arange(len(lh_p)), lh_p, linewidth=0.5); axs[1].set_yscale("log"); axs[1].set_ylim([1e-5, 1e-1]); axs[1].set_title("Data loss")
                    axs[2].plot(np.arange(len(lh_v)), lh_v, linewidth=0.5); axs[2].set_yscale("log"); axs[2].set_ylim([1e-5, 1e-1]); axs[2].set_title("Physics loss")
            axs[0].legend(loc="lower left", prop={"size":12})
            print()
    else:
        params = lh_filename.split("_")[-3:-1]
        if params[1]=='16':
            j = 3
        elif params[1]=='2':
            j = 0
        elif params[1]=='4':
            j = 1
        elif params[1]=='8':
            j = 2
        #fig.suptitle(f"FNO f"$L$={params[0]} $d_v$={params[1]}")
        with open(lh_filename, 'rb') as loss_file:
            lhs = pickle.load(loss_file)
            with open(metrics_filename, 'rb') as metrics_file:
                metrics = pickle.load(metrics_file)
                print(params)
                for k, (lh, lh_v) in enumerate(zip(lhs['train'], lhs['validation'])):
                    print(metrics["R2"])
                    if k>=8:
                        continue
                    r2 = metrics["R2"][k].item()
                    axs[0,j].plot(np.arange(len(lh)), lh, linewidth=0.5, label="{:.2f}".format(r2))
                    axs[0,j].set(yscale="log", ylim=[1e-6, 1e-1], title=f"$L$={params[0]} $d_v$={params[1]}", ylabel="Train MSE")
                    axs[1,j].plot(np.arange(len(lh_v)), lh_v, linewidth=0.5)
                    axs[1,j].set(yscale="log", ylim=[1e-6, 1e-1], xlabel="Epoch", ylabel="Validation MSE")
                print(metrics)
            axs[0,j].legend(loc="lower left", prop={"size":12})
        fig.tight_layout()
        #fig.savefig("training_loss_L{}_adjoint.pdf".format(params[0]))
"""
for filename in metrics_filenames:
    with open(filename, 'rb') as infile:
        metrics = pickle.load(infile)
        print(filename.split("_")[-4:-1])
        print("R2 =",*("{:.2g}".format(r2.item()) for r2 in metrics["R2"]))
        print("loss =",torch.mean(torch.tensor(metrics["test_loss"])[:,1]))
        print("t =", *("{:.3g}".format(t) for t in metrics["training_times"]))
        print()
        #break
        #s = 0
        #for r2 in metrics["R2"]:
        #    s+=r2.item()
        #print(s/len(metrics["R2"]))
#with open(loss_history_filename, 'rb') as infile:
#    loss_histories = pickle.load(infile)
"""
