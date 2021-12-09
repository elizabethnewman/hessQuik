import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from peaks import peaks


#%% training functions

def train_one_epoch(net, x, y, optimizer, batch_size=5, loss_weights=(1.0, 1.0, 1.0)):
    net.train()
    n_train = x.shape[0]
    b = batch_size
    n_batch = n_train // b

    idx = torch.randperm(n_train)

    running_loss, running_loss_f, running_loss_df, running_loss_d2f = 0.0, 0.0, 0.0, 0.0
    for i in range(n_batch):
        xb, yb = x[idx[i * b:(i + 1) * b]], y[idx[i * b:(i + 1) * b]]

        fb, dfb, d2fb = net(xb, do_gradient=True, do_Hessian=True)

        yb_hat = torch.cat((fb.reshape(b, -1), dfb.reshape(b, -1), d2fb.reshape(b, -1)[:, (0, 1, 3)]), dim=1)

        loss_f = (0.5 / b) * torch.norm(yb_hat[:, 0] - yb[:, 0]) ** 2
        loss_df = (0.5 / b) * torch.norm(yb_hat[:, 1:3] - yb[:, 1:3]) ** 2
        loss_d2f = (0.5 / b) * torch.norm(yb_hat[:, 3:] - yb[:, 3:]) ** 2
        loss = loss_weights[0] * loss_f + loss_weights[1] * loss_df + loss_weights[2] * loss_d2f

        # store running loss
        running_loss_f += loss_f.item()
        running_loss_df += loss_df.item()
        running_loss_d2f += loss_d2f.item()
        running_loss += loss.item()

        # update network weights
        loss.backward()
        optimizer.step()

    return running_loss, running_loss_f, running_loss_df, running_loss_d2f


def test(net, x, y, loss_weights=(1.0, 1.0, 1.0)):
    net.eval()

    n = x.shape[0]
    f0, df0, d2f0 = net(x, do_gradient=True, do_Hessian=True)
    y_hat = torch.cat((f0.reshape(n, -1), df0.reshape(n, -1), d2f0.reshape(n, -1)[:, (0, 1, 3)]), dim=1)

    loss_f = (0.5 / n) * torch.norm(y_hat[:, 0] - y[:, 0]) ** 2
    loss_df = (0.5 / n) * torch.norm(y_hat[:, 1:3] - y[:, 1:3]) ** 2
    loss_d2f = (0.5 / n) * torch.norm(y_hat[:, 3:] - y[:, 3:]) ** 2
    loss = loss_weights[0] * loss_f + loss_weights[1] * loss_df + loss_weights[2] * loss_d2f

    return loss.item(), loss_f.item(), loss_df.item(), loss_d2f.item()

# define device
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

#%% Create data
n_train = 1000
n_val = 100
n_test = 100
x = -3 + 6 * torch.rand(n_train + n_val + n_test, 2, device=device)
f, df, d2f = peaks(x, do_gradient=True, do_Hessian=True)
y = torch.cat((f, df.view(x.shape[0], -1), d2f.view(x.shape[0], -1)[:, (0, 1, 3)]), dim=1)

# split
idx = torch.randperm(n_train + n_val + n_test)
x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
x_val, y_val = x[idx[n_train:n_train+n_val]], y[idx[n_train:n_train+n_val]]
x_test, y_test = x[idx[n_train+n_val:]], y[idx[n_train+n_val:]]


#%% create network
width = 8
depth = 4
net = net.NN(lay.singleLayer(2, width, act=act.softplusActivation()),
             net.resnetNN(width, depth, h=0.25, act=act.softplusActivation()),
             lay.singleLayer(width, 1, act=act.softplusActivation()))

# define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


#%% train!

headers = ('', '', '|', 'running', '', '', '', '|', 'train', '', '', '', '|', 'valid', '', '', '')

tmp = ('|', 'loss', 'loss_f', 'loss_df', 'loss_df2')
printouts = ('epoch', '|K1|') + 3 * tmp

tmp = '{:<2s}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}'
printouts_frmt = '{:<15d}{:<15.4e}' + 3 * tmp

tmp = '{:<2s}{:<15s}{:<15s}{:<15s}{:<15s}'
print(('{:<15s}{:<15s}' + 3 * tmp).format(*headers))
print(('{:<15s}{:<15s}' + 3 * tmp).format(*printouts))

max_epochs = 20
batch_size = 5

loss_weights = (0, 0, 1)
for epoch in range(max_epochs):
    running_loss = train_one_epoch(net, x_train, y_train, optimizer, batch_size, loss_weights=loss_weights)

    # test
    loss_train = test(net, x_train, y_train, loss_weights=loss_weights)
    loss_val = test(net, x_val, y_val, loss_weights=loss_weights)

    his = (epoch, torch.norm(net[0].K).item()) + ('|',) + running_loss + ('|',) + loss_train + ('|',) + loss_val
    print(printouts_frmt.format(*his))

# overall performance on test data
loss_test = test(net, x_test, y_test)
print('Test Loss: %0.4e' % loss_test[0])
