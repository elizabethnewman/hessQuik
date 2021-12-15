import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from peaks import peaks
from time import time
import numpy as np
import matplotlib.pyplot as plt


# %% training functions
def train_one_epoch(f, x, y, optimizer, batch_size=5, do_gradient=True, do_Hessian=True, loss_weights=(1.0, 1.0, 1.0)):
    f.train()
    n = x.shape[0]
    b = batch_size
    n_batch = n_train // b

    loss_f, loss_df, loss_d2f = torch.zeros(1), torch.zeros(1), torch.zeros(1)

    idx = torch.randperm(n)

    running_loss, running_loss_f, running_loss_df, running_loss_d2f = 0.0, 0.0, 0.0, 0.0
    for i in range(n_batch):
        idxb = idx[i * b:(i + 1) * b]
        xb, yfb, ydfb, yd2fb = x[idxb], y['f'][idxb], y['df'][idxb], y['d2f'][idxb]

        optimizer.zero_grad()
        fb, dfb, d2fb = f(xb, do_gradient=do_gradient, do_Hessian=do_Hessian)

        loss_f = (0.5 / b) * torch.norm(fb - yfb) ** 2

        if dfb is not None:
            loss_df = (0.5 / b) * torch.norm(dfb - ydfb) ** 2
        if d2fb is not None:
            loss_d2f = (0.5 / b) * torch.norm(d2fb - yd2fb) ** 2

        loss = loss_weights[0] * loss_f + loss_weights[1] * loss_df + loss_weights[2] * loss_d2f

        # store running loss
        running_loss_f += b * loss_f.item()
        running_loss_df += b * loss_df.item()
        running_loss_d2f += b * loss_d2f.item()
        running_loss += b * loss.item()

        # update network weights
        loss.backward()
        optimizer.step()

    return running_loss / n, running_loss_f / n, running_loss_df / n, running_loss_d2f / n


def test(f, x, y, do_gradient=True, do_Hessian=True, loss_weights=(1.0, 1.0, 1.0)):
    f.eval()

    loss_f, loss_df, loss_d2f = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    with torch.no_grad():
        n = x.shape[0]
        f0, df0, d2f0 = f(x, do_gradient=do_gradient, do_Hessian=do_Hessian)

        loss_f = (0.5 / n) * torch.norm(f0 - y['f']) ** 2

        if df0 is not None:
            loss_df = (0.5 / n) * torch.norm(df0 - y['df']) ** 2

        if d2f0 is not None:
            loss_d2f = (0.5 / n) * torch.norm(d2f0 - y['d2f']) ** 2

        loss = loss_weights[0] * loss_f + loss_weights[1] * loss_df + loss_weights[2] * loss_d2f

    return loss.item(), loss_f.item(), loss_df.item(), loss_d2f.item()


# define device
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# %% Create data
n_train = 1000
n_val = 50
n_test = 50
x = -3 + 6 * torch.rand(n_train + n_val + n_test, 2, device=device)
yc, dy, d2y = peaks(x, do_gradient=True, do_Hessian=True)
y = {'f': yc, 'df': dy, 'd2f': d2y}

# split
idx = torch.randperm(n_train + n_val + n_test)
x_train, y_train = x[idx[:n_train]], {'f': yc[idx[:n_train]], 'df': dy[idx[:n_train]], 'd2f': d2y[idx[:n_train]]}
x_val, y_val = x[idx[n_train:n_train + n_val]],  {'f': yc[idx[n_train:n_train + n_val]], 'df': dy[idx[n_train:n_train + n_val]], 'd2f': d2y[idx[n_train:n_train + n_val]]}
x_test, y_test = x[idx[n_train + n_val:]], {'f': yc[idx[n_train + n_val:]], 'df': dy[idx[n_train + n_val:]], 'd2f': d2y[idx[n_train + n_val:]]}

# %% create network
width = 16
depth = 4
f = net.NN(lay.singleLayer(2, width, act=act.tanhActivation()),
           net.resnetNN(width, depth, h=1.0, act=act.tanhActivation()),
           lay.singleLayer(width, 1, act=act.identityActivation()))

# define optimizer
optimizer = torch.optim.Adam(f.parameters(), lr=1e-3)

# %% train!
headers = ('', '', '', '|', 'running', '', '', '', '|', 'train', '', '', '', '|', 'valid', '', '', '')

tmp = ('|', 'loss', 'loss_f', 'loss_df', 'loss_df2')
printouts = ('epoch', '|K1|', 'time') + 3 * tmp

tmp = '{:<2s}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}'
printouts_frmt = '{:<15d}{:<15.4e}{:<15.4f}' + 3 * tmp

tmp = '{:<2s}{:<15s}{:<15s}{:<15s}{:<15s}'
print(('{:<15s}{:<15s}{:<15s}' + 3 * tmp).format(*headers))
print(('{:<15s}{:<15s}{:<15s}' + 3 * tmp).format(*printouts))

max_epochs = 50
batch_size = 5
loss_weights = (1.0, 1.0, 1.0)
do_gradient = False
do_Hessian = False

# initial evaluation
loss_train = test(f, x_train, y_train, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
loss_val = test(f, x_val, y_val, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)


his = (-1, torch.norm(f[0].K.data).item(), 0.0) + ('|',) + (4 * (0,)) + ('|',) + loss_train + ('|',) + loss_val
print(printouts_frmt.format(*his))

His = np.array([x for x in his if not (x == '|')]).reshape(1, -1)


for epoch in range(max_epochs):
    t0 = time()
    running_loss = train_one_epoch(f, x_train, y_train, optimizer, batch_size,
                                   do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
    t1 = time()

    # test
    loss_train = test(f, x_train, y_train, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
    loss_val = test(f, x_val, y_val, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)

    his = (epoch, torch.norm(f[0].K.data).item(), t1 - t0) + ('|',) + running_loss + ('|',) + loss_train + ('|',) + loss_val
    print(printouts_frmt.format(*his))

    His = np.concatenate((His, np.array([x for x in his if not (x == '|')]).reshape(1, -1)), axis=0)

# overall performance on test data
loss_test = test(f, x_test, y_test, loss_weights=loss_weights)
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

plt.figure()
linewidth = 3
plt.semilogy(His[:, 0], His[:, 8], linewidth=linewidth, label='f')

if do_gradient:
    plt.semilogy(His[:, 0], His[:, 9], linewidth=linewidth, label='df')

if do_Hessian:
    plt.semilogy(His[:, 0], His[:, 10], linewidth=linewidth, label='d2f')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim([1e-3, 1e1])
plt.legend()
plt.show()

