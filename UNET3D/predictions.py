import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def plot_predictions(model, device, input, target):
    #input is (b,c,w,h,d)
    with torch.no_grad:
        pred = model(input)
    pred = pred.numpy()
    target = target.numpy()
    input = input.numpy()

    fig = plt.figure()
    plot_titles = ['t1', 't1c', 't2', 'flair', 'gt', 'pred']
    n_slices = 5
    slices = np.linspace(20,140, n_slices)
    for idx_i, slice in enumerate(slices):
        for idx_j, title in enumerate(plot_titles):
            plt.subplot(n_slices*len(plot_titles), idx_j % len(plot_titles + idx_i * len(plot_titles + 1)))
            if idx_j <= 3:
                plt.imshow(input[0,idx_j,:,:,slice])
            if idx_j == 4:
                plt.imshow(target[0,idx_j,:,:,slice])
            else:
                plt.imshow(pred[0,idx_j,:,:,slice])
            if idx_i == 0:
                plt.title(title)
    return fig

