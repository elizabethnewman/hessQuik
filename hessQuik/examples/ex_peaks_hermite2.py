import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from hessQuik.utils import peaks, train_one_epoch, test, print_headers
import time
import numpy as np

torch.manual_seed(42)

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

n_train = 1000      # number of training points
n_val = 100         # number of validation points
n_test = 100        # number of testing points

# generate data
x = -3 + 6 * torch.rand(n_train + n_val + n_test, 2, device=device)
yc, dy, d2y = peaks(x, do_gradient=True, do_Hessian=True)
y = torch.cat((yc, dy.view(yc.shape[0], -1), d2y.view(yc.shape[0], -1)), dim=1)

# shuffle and split data
idx = torch.randperm(n_train + n_val + n_test)
x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
x_val, y_val = x[idx[n_train:n_train + n_val]], y[idx[n_train:n_train + n_val]]
x_test, y_test = x[idx[n_train + n_val:]], y[idx[n_train + n_val:]]

#%%

width = 16
depth = 4
f = net.NN(lay.singleLayer(2, width, act=act.tanhActivation()),
           net.resnetNN(width, depth, h=0.5, act=act.tanhActivation()),
           lay.singleLayer(width, 1, act=act.identityActivation())).to(device)

# Pytorch optimizer for the network weights
optimizer = torch.optim.Adam(f.parameters(), lr=1e-3)

#%%
# training parameters
max_epochs = 5
batch_size = 5
loss_weights = (0.0, 0.0, 1.0)
do_gradient = True
do_Hessian = True

headers, printouts, printouts_frmt = print_headers(do_gradient=do_gradient, do_Hessian=do_Hessian,
                                                   loss_weights=loss_weights)


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
for epoch in range(max_epochs):
    t0 = time.perf_counter()
    running_loss = train_one_epoch(f, x_train, y_train, optimizer, batch_size,
                                   do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
    t1 = time.perf_counter()

    # test
    loss_train = test(f, x_train, y_train, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
    loss_val = test(f, x_val, y_val, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)

    his_iter = (epoch, t1 - t0) + ('|',) + running_loss + ('|',) + loss_train + ('|',) + loss_val
    print(printouts_frmt.format(*his_iter))

    # store history
    his = np.concatenate((his, np.array([x for x in his_iter if not (x == '|')]).reshape(1, -1)), axis=0)

# ---------------------------------------------------------------------------- #
# overall performance on test data
loss_test = test(f, x_test, y_test, do_gradient=do_gradient, do_Hessian=do_Hessian, loss_weights=loss_weights)
print('Test Loss: %0.4e' % loss_test[0])

