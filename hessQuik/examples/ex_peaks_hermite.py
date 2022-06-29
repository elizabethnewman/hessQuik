import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from hessQuik.utils import peaks, train_one_epoch, test, print_headers
from time import time
import numpy as np
import matplotlib.pyplot as plt
import time

# define device
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# %% Create data
n_train = 1000
n_val = 50
n_test = 50
x = -3 + 6 * torch.rand(n_train + n_val + n_test, 2, device=device)
yc, dy, d2y = peaks(x, do_gradient=True, do_Hessian=True)
y = torch.cat((yc, dy.view(yc.shape[0], -1), d2y.view(yc.shape[0], -1)), dim=1)

# split
idx = torch.randperm(n_train + n_val + n_test)
x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
x_val, y_val = x[idx[n_train:n_train + n_val]], y[idx[n_train:n_train + n_val]]
x_test, y_test = x[idx[n_train + n_val:]], y[idx[n_train + n_val:]]

# %% create network
width = 16
depth = 4
f = net.NN(lay.singleLayer(2, width, act=act.tanhActivation()),
           net.resnetNN(width, depth, h=1.0, act=act.tanhActivation()),
           lay.singleLayer(width, 1, act=act.identityActivation()))

# define optimizer
optimizer = torch.optim.Adam(f.parameters(), lr=1e-3)

# %% train!

# training parameters
max_epochs = 50
batch_size = 5
loss_weights = (1.0, 1.0, 1.0)
do_gradient = True
do_Hessian = True

# get printouts
headers, printouts_str, printouts_frmt = print_headers(do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)

# ---------------------------------------------------------------------------- #
# initial evaluation
loss_train = test(f, x_train, y_train, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
loss_val = test(f, x_val, y_val, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)

n_loss = 2 + do_gradient + do_Hessian
his_iter = (-1, 0.0) + ('|',) + (n_loss * (0,)) + ('|',) + loss_train + ('|',) + loss_val
print(printouts_frmt.format(*his_iter))

# store history
his = np.array([x for x in his_iter if not (x == '|')]).reshape(1, -1)

# ---------------------------------------------------------------------------- #
# main iteration

log_interval = 5 # how often printouts appear
for epoch in range(max_epochs):
    t0 = time.perf_counter()
    running_loss = train_one_epoch(f, x_train, y_train, optimizer, batch_size,
                                   do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
    t1 = time.perf_counter()

    # test
    loss_train = test(f, x_train, y_train, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
    loss_val = test(f, x_val, y_val, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)

    his_iter = (epoch, t1 - t0) + ('|',) + running_loss + ('|',) + loss_train + ('|',) + loss_val

    if epoch % log_interval == 0:
        print(printouts_frmt.format(*his_iter))

    # store history
    his = np.concatenate((his, np.array([x for x in his_iter if not (x == '|')]).reshape(1, -1)), axis=0)

# ---------------------------------------------------------------------------- #
# overall performance on test data
loss_test = test(f, x_test, y_test, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
print('Test Loss: %0.4e' % loss_test[0])

#%%
xy = torch.arange(-3, 3, 0.25, dtype=torch.float32)
grid_x, grid_y = torch.meshgrid(xy, xy)

grid_data = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)

# approximation
c_true = peaks(grid_data)[0].view(grid_x.shape).numpy()
c_pred = f.forward(grid_data)[0].detach().view(grid_x.shape).numpy()

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
p = ax.imshow(c_pred.reshape(grid_x.shape), vmin=c_true.min(), vmax=c_true.max())
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

plt.show()

# convergence plots
plt.figure()
linewidth = 3
idx = [idx for idx, n in enumerate(np.array([x for x in printouts_str if not (x == '|')])) if n == 'loss_f'][1]

plt.semilogy(his[:, 0], his[:, idx], linewidth=linewidth, label='f')

if do_gradient:
  plt.semilogy(his[:, 0], his[:, idx + 1], linewidth=linewidth, label='df')

if do_Hessian:
  plt.semilogy(his[:, 0], his[:, idx + 2], linewidth=linewidth, label='d2f')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training loss')
plt.legend()
plt.show()
