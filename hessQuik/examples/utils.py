import torch
import numpy as np
import matplotlib.pyplot as plt


def network_imshow(net, grid_x, grid_y, c_true):

    grid_data = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)

    # approximation
    c_pred = net.forward(grid_data).detach().view(grid_x.shape).numpy()

    print('relative error = ',
          np.linalg.norm(c_pred.reshape(-1) - c_true.reshape(-1)) / np.linalg.norm(c_true.reshape(-1)))

    # image plots
    fig, axs = plt.subplots(2, 2)
    ax = axs[0, 0]
    p = ax.imshow(c_true.reshape(grid_x.shape))
    ax.axis('off')
    ax.set_title('true')
    fig.colorbar(p, ax=ax, aspect=10)

    ax = axs[0, 1]
    p = ax.imshow(c_pred.reshape(grid_x.shape))
    ax.axis('off')
    ax.set_title('approx')
    fig.colorbar(p, ax=ax, aspect=10)

    ax = axs[1, 0]
    p = ax.imshow(np.abs(c_pred - c_true).reshape(grid_x.shape))
    fig.colorbar(p, ax=ax, aspect=10)
    ax.axis('off')
    ax.set_title('abs. diff.')

    ax = axs[1, 1]
    ax.axis('off')
