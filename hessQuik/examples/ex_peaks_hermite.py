import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from peaks import peaks


# %% training functions
def train_one_epoch(f, x, y, optimizer, batch_size=5, loss_weights=(1.0, 1.0, 1.0)):
    f.train()
    n_train = x.shape[0]
    b = batch_size
    n_batch = n_train // b

    idx = torch.randperm(n_train)

    running_loss, running_loss_f, running_loss_df, running_loss_d2f = 0.0, 0.0, 0.0, 0.0
    for i in range(n_batch):
        idxb = idx[i * b:(i + 1) * b]
        xb, yb = x[idxb], y[idxb]

        fb, dfb, d2fb = f(xb, do_gradient=True, do_Hessian=True)

        loss_f = (0.5 / b) * torch.norm(fb.view(b, -1) - yb[:, 0]) ** 2
        loss_df = (0.5 / b) * torch.norm(dfb.view(b, -1) - yb[:, 1:3]) ** 2
        loss_d2f = (0.5 / b) * torch.norm(d2fb.view(b, -1) - yb[:, 3:]) ** 2
        loss = loss_weights[0] * loss_f + loss_weights[1] * loss_df + loss_weights[2] * loss_d2f

        # store running loss
        running_loss_f += loss_f.item()
        running_loss_df += loss_df.item()
        running_loss_d2f += b * loss_d2f.item()
        running_loss += b * loss.item()

        # update network weights
        loss.backward()
        optimizer.step()

    return running_loss / n_train, running_loss_f / n_train, running_loss_df / n_train, running_loss_d2f / n_train


def test(f, x, y, loss_weights=(1.0, 1.0, 1.0)):
    f.eval()

    with torch.no_grad():
        n = x.shape[0]
        f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)
        # y_hat = torch.cat((f0.reshape(n, -1), df0.reshape(n, -1), d2f0.reshape(n, -1)[:, (0, 1, 3)]), dim=1)

        loss_f = (0.5 / n) * torch.norm(f0.view(n, -1) - y[:, 0]) ** 2
        loss_df = (0.5 / n) * torch.norm(df0.view(n, -1) - y[:, 1:3]) ** 2
        loss_d2f = (0.5 / n) * torch.norm(d2f0.view(n, -1) - y[:, 3:]) ** 2
        loss = loss_weights[0] * loss_f + loss_weights[1] * loss_df + loss_weights[2] * loss_d2f

    return loss.item(), loss_f.item(), loss_df.item(), loss_d2f.item()


# define device
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# %% Create data
n_train = 1000
n_val = 100
n_test = 100
x = -3 + 6 * torch.rand(n_train + n_val + n_test, 2, device=device)
yc, dy, d2y = peaks(x, do_gradient=True, do_Hessian=True)
y = torch.cat((yc, dy.view(x.shape[0], -1), d2y.view(x.shape[0], -1)), dim=1)

# split
idx = torch.randperm(n_train + n_val + n_test)
x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
x_val, y_val = x[idx[n_train:n_train + n_val]], y[idx[n_train:n_train + n_val]]
x_test, y_test = x[idx[n_train + n_val:]], y[idx[n_train + n_val:]]

# %% create network
width = 32
depth = 8
f = net.NN(lay.singleLayer(2, width, act=act.tanhActivation()),
           net.resnetNN(width, depth, h=1.0, act=act.tanhActivation()),
           lay.singleLayer(width, 1, act=act.tanhActivation()))

# define optimizer
optimizer = torch.optim.Adam(f.parameters(), lr=1e-3)

# %% train!
headers = ('', '', '|', 'running', '', '', '', '|', 'train', '', '', '', '|', 'valid', '', '', '')

tmp = ('|', 'loss', 'loss_f', 'loss_df', 'loss_df2')
printouts = ('epoch', '|K1|') + 3 * tmp

tmp = '{:<2s}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}'
printouts_frmt = '{:<15d}{:<15.4e}' + 3 * tmp

tmp = '{:<2s}{:<15s}{:<15s}{:<15s}{:<15s}'
print(('{:<15s}{:<15s}' + 3 * tmp).format(*headers))
print(('{:<15s}{:<15s}' + 3 * tmp).format(*printouts))

max_epochs = 50
batch_size = 5
loss_weights = (0, 0, 1.0)

# initial evaluation
loss_train = test(f, x_train, y_train, loss_weights=loss_weights)
loss_val = test(f, x_val, y_val, loss_weights=loss_weights)

his = (-1, torch.norm(f[0].K).item()) + ('|',) + (4 * (0,)) + ('|',) + loss_train + ('|',) + loss_val
print(printouts_frmt.format(*his))


for epoch in range(max_epochs):
    running_loss = train_one_epoch(f, x_train, y_train, optimizer, batch_size, loss_weights=loss_weights)

    # test
    loss_train = test(f, x_train, y_train, loss_weights=loss_weights)
    loss_val = test(f, x_val, y_val, loss_weights=loss_weights)

    his = (epoch, torch.norm(f[0].K).item()) + ('|',) + running_loss + ('|',) + loss_train + ('|',) + loss_val
    print(printouts_frmt.format(*his))

# overall performance on test data
loss_test = test(f, x_test, y_test, loss_weights=loss_weights)
print('Test Loss: %0.4e' % loss_test[0])
