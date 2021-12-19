import torch


def train_one_epoch(f, x, y, optimizer, batch_size=5, do_gradient=True, do_Hessian=True, loss_weights=(1.0, 1.0, 1.0)):
    f.train()
    n = x.shape[0]
    b = batch_size
    n_batch = n // b

    (loss_f, loss_df, loss_d2f) = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    running_loss, running_loss_f, running_loss_df, running_loss_d2f = (0.0, 0.0, 0.0 if do_gradient else None,
                                                                       0.0 if do_Hessian else None)

    # shuffle
    idx = torch.randperm(n)

    for i in range(n_batch):
        idxb = idx[i * b:(i + 1) * b]
        xb, yb = x[idxb], y[idxb]

        optimizer.zero_grad()
        fb, dfb, d2fb = f(xb, do_gradient=do_gradient, do_Hessian=do_Hessian)

        loss_f = (0.5 / b) * torch.norm(fb - yb[:, 0].view_as(fb)) ** 2
        loss = loss_weights[0] * loss_f
        running_loss_f += b * loss_f.item()

        if do_gradient:
            loss_df =  (0.5 / b) * torch.norm(dfb - yb[:, 1:xb.shape[1]+1].view_as(dfb)) ** 2
            loss = loss + loss_weights[1] * loss_df
            running_loss_df += b * loss_df.item()
        if do_Hessian:
            loss_d2f = (0.5 / b) * torch.norm(d2fb - yb[:, xb.shape[1]+1:].view_as(d2fb)) ** 2
            loss = loss + loss_weights[2] * loss_d2f
            running_loss_d2f += b * loss_d2f.item()

        running_loss += b * loss.item()

        # update network weights
        loss.backward()
        optimizer.step()

    output = (running_loss / n, running_loss_f / n)
    if do_gradient:
        output += (running_loss_df / n,)
        if do_Hessian:
            output += (running_loss_d2f / n,)

    return output


def test(f, x, y, do_gradient=True, do_Hessian=True, loss_weights=(1.0, 1.0, 1.0)):
    f.eval()

    (loss_f, loss_df, loss_d2f) = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    with torch.no_grad():
        n = x.shape[0]
        f0, df0, d2f0 = f(x, do_gradient=do_gradient, do_Hessian=do_Hessian)

        loss_f = (0.5 / n) * torch.norm(f0 - y[:, 0].view_as(f0)) ** 2
        loss = loss_weights[0] * loss_f

        if do_gradient:
            loss_df = (0.5 / n) * torch.norm(df0 - y[:, 1:x.shape[1]+1].view_as(df0)) ** 2
            loss +=  loss_weights[1] * loss_df

        if do_Hessian:
            loss_d2f = (0.5 / n) * torch.norm(d2f0 - y[:, x.shape[1]+1:].view_as(d2f0)) ** 2
            loss += loss_weights[2] *  loss_d2f

    output = (loss.item(), loss_f.item())
    if do_gradient:
        output += (loss_df.item(),)
        if do_Hessian:
            output += (loss_d2f.item(),)

    return output


def print_headers(do_gradient=True, do_Hessian=True, verbose=True, loss_weights=(1.0, 1.0, 1.0)):

    loss_printouts = ('loss', 'loss_f')
    if do_gradient or do_Hessian:
        loss_printouts += ('loss_df',)
        if do_Hessian:
            loss_printouts += ('loss_d2f',)
    n_loss = len(loss_printouts)

    headers = (('', '', '|', 'running',) + (n_loss - 1) * ('',) + ('|', 'train',)
               + (n_loss - 1) * ('',) + ('|', 'valid',) + (n_loss - 1) * ('',))
    weights = (('', '', '|', 'weights',) + loss_weights[:n_loss - 1] + ('|', '',)
               + loss_weights[:n_loss - 1] + ('|', '',) + loss_weights[:n_loss - 1])

    printouts = ('epoch', 'time') + 3 * (('|',) + loss_printouts)
    printouts_frmt = '{:<15d}{:<15.4f}' + 3 * ('{:<2s}' + n_loss * '{:<15.4e}')

    if verbose:
        print(('{:<15s}{:<15s}' + 3 * ('{:<2s}' + n_loss * '{:<15s}')).format(*headers))
        print(('{:<15s}{:<15s}' + 3 * ('{:<2s}{:<15s}' + (n_loss - 1) * '{:<15.2e}')).format(*weights))
        print(('{:<15s}{:<15s}' + 3 * ('{:<2s}' + n_loss * '{:<15s}')).format(*printouts))

    return printouts_frmt
