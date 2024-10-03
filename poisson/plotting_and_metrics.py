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

model_name = "FNO" # "DON"
problem = "state" # "adjoint"


model_list_filenames = glob.glob('.\{}_models\models_list_*.pkl'.format(model_name))
metrics_filenames = glob.glob('.\{}_models\metrics_*.pkl'.format(model_name))
loss_history_filenames = glob.glob('.\{}_models\loss_history_*'.format(model_name))

#with open(model_list_filename, 'rb') as infile:
#    model_list = pickle.load(infile)
for lh_filename, metrics_filename in zip(loss_history_filenames, metrics_filenames):
    fig, axs = plt.subplots(1,2 + (model_name=="DON"), figsize=(12,5))
    params = lh_filename.split("_")[-4:-1]
    fig.suptitle(params[0] + "," + params[1] + "," + params[2])
    with open(lh_filename, 'rb') as loss_file:
        lhs = pickle.load(loss_file)
        with open(metrics_filename, 'rb') as metrics_file:
            metrics = pickle.load(metrics_file)
            print(params)
            if model_name=="DON":
                for i, (lh_d, lh_p, lh, lh_v) in enumerate(zip(lhs['data'], lhs['physics'], lhs['total'], lhs['validation'])):
                    if i>=10:
                        break
                    r2 = metrics["R2"][i].item()
                    axs[0].plot(np.arange(len(lh))[:], lh[:], linewidth=0.5, label="{:.2f}".format(r2)); axs[0].set_yscale("log"); axs[0].set_ylim([1e-2, 1e1]); axs[0].set_title("Total loss")
                    axs[1].plot(np.arange(len(lh_v))[:], lh_v[:], linewidth=0.5); axs[1].set_yscale("log"); axs[1].set_ylim([1e-2, 1e1])
                    axs[2].plot(np.arange(len(lh_d))[:], lh_d[:], linewidth=0.5, linestyle="--")
                    axs[2].plot(np.arange(len(lh_p))[:], lh_p[:], linewidth=0.5); axs[2].set_yscale("log"); axs[2].set_ylim([1e-4, 1e0]); axs[2].set_title("Data & physics loss")
            else:
                for i, (lh, lh_v) in enumerate(zip(lhs['train'], lhs['validation'])):
                    if i>=10:
                        break
                    r2 = metrics["R2"][i].item()
                    axs[0].plot(np.arange(len(lh)), lh, linewidth=0.5, label="{:.2f}".format(r2)); axs[0].set_yscale("log"); axs[0].set_ylim([1e-5, 1e0]); axs[0].set_title("Train loss")
                    axs[1].plot(np.arange(len(lh_v)), lh_v, linewidth=0.5); axs[1].set_yscale("log"); axs[1].set_ylim([1e-5, 1e0]); axs[1].set_title("Validation loss")
            print(metrics["training_times"])
            print(metrics["test_loss"])
            print(metrics["R2"])
        axs[0].legend(loc="lower left")
        print()