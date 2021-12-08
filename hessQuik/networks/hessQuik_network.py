import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import hessian


class NN(nn.Sequential):
    """
    Forward propagation through network composed of forward Hessian layers
    """
    def __init__(self, *args):
        super(NN, self).__init__(*args)

    def forward(self, x, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None, reverse_mode=False):
        for module in self:
            x, dudx, d2ud2x = module(x, do_gradient=do_gradient, do_Hessian=do_Hessian, dudx=dudx, d2ud2x=d2ud2x,
                                     reverse_mode=reverse_mode)
        return x, dudx, d2ud2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        for i in range(len(self) - 1, -1, -1):
            dgdf, d2gd2f = self[i].backward(do_Hessian=do_Hessian, dgdf=dgdf, d2gd2f=d2gd2f)
        return dgdf, d2gd2f


class NNPytorchAD(nn.Module):

    def __init__(self, net):
        super(NNPytorchAD, self).__init__()
        self.net = net
        self.ctx = None

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (df, d2f) = (None, None)
        f, *_ = self.net(x)

        if reverse_mode:
            self.ctx = (f, x)

        if not reverse_mode and (do_gradient or do_Hessian):
            f = f.view(x.shape[0], -1)
            df = []
            for j in range(f.shape[1]):
                df.append(grad(f[:, j].sum(), x, create_graph=True, retain_graph=True)[0])
            df = torch.stack(df, dim=2)

            if do_Hessian:
                df = df.reshape(x.shape[0], -1)
                d2f = []
                for j in range(df.shape[1]):
                    d2f.append(grad(df[:, j].sum(), x, create_graph=True, retain_graph=True)[0])
                d2f = torch.stack(d2f, dim=2)
                d2f = d2f.reshape(x.shape[0], x.shape[1], x.shape[1], -1).squeeze(-1)
                if d2f.dim() < 4:
                    d2f = d2f.unsqueeze(-1)

            df = df.reshape(x.shape[0], x.shape[1], -1).squeeze(-1)

            if df.dim() < 3:
                df = df.unsqueeze(-1)

        return f, df, d2f

    def backward(self, do_Hessian=False):
        f, x = self.ctx

        f = f.view(x.shape[0], -1)
        df, d2f = [], None
        for j in range(f.shape[1]):
            df.append(grad(f[:, j].sum(), x, create_graph=True, retain_graph=True)[0])
        df = torch.stack(df, dim=2)

        if do_Hessian:
            df = df.reshape(x.shape[0], -1)
            d2f = []
            for j in range(df.shape[1]):
                d2f.append(grad(df[:, j].sum(), x, create_graph=True, retain_graph=True)[0])
            d2f = torch.stack(d2f, dim=2)
            d2f = d2f.reshape(x.shape[0], x.shape[1], x.shape[1], -1).squeeze(-1)
            if d2f.dim() < 4:
                d2f = d2f.unsqueeze(-1)

        df = df.reshape(x.shape[0], x.shape[1], -1).squeeze(-1)

        if df.dim() < 3:
            df = df.unsqueeze(-1)

        return df, d2f


class NNPytorchHessian(nn.Module):

    def __init__(self, net):
        super(NNPytorchHessian, self).__init__()
        self.net = net
        self.ctx = None

    def forward(self, x, do_gradient=False, do_Hessian=False, reverse_mode=False):
        (df, d2f) = (None, None)
        f, *_ = self.net(x, do_gradient=False, do_Hessian=False, reverse_mode=reverse_mode)

        if f.squeeze().ndim > 1:
            raise ValueError(type(self), " must have scalar outputs per example")

        if reverse_mode:
            self.ctx = (f, x)

        if not reverse_mode and (do_gradient or do_Hessian):
            df = grad(f.sum(), x)[0]

            if do_Hessian:
                d2f = hessian(lambda x: self.net(x)[0].sum(), x).sum(dim=2)

        return f, df, d2f

    def backward(self, do_Hessian=False):
        d2f = None
        f, x = self.ctx
        df = grad(f.sum(), x)[0]

        if do_Hessian:
            d2f = hessian(lambda x: self.net(x)[0].sum(), x).sum(dim=2)

        return df, d2f

