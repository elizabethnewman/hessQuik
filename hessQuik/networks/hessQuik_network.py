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

    def dim_input(self):
        return self[0].dim_input()

    def dim_output(self):
        return self[-1].dim_output()

    def setup_forward_mode(self, **kwargs):
        if not ('forward_mode' in kwargs.keys()):
            if self.dim_input() < self.dim_output():
                forward_mode = True  # compute the derivatives in forward mode
            else:
                forward_mode = False  # store necessary info, but do not compute derivatives until backward call
            kwargs['forward_mode'] = forward_mode

        return kwargs['forward_mode']

    def forward(self, x, do_gradient=False, do_Hessian=False, dudx=None, d2ud2x=None, **kwargs):
        forward_mode = self.setup_forward_mode(**kwargs)

        for module in self:
            x, dudx, d2ud2x = module(x, do_gradient=do_gradient, do_Hessian=do_Hessian, dudx=dudx, d2ud2x=d2ud2x,
                                     forward_mode=True if forward_mode is True else None)

        if (do_gradient or do_Hessian) and forward_mode is False:
            dudx, d2ud2x = self.backward(do_Hessian=do_Hessian)

        return x, dudx, d2ud2x

    def backward(self, do_Hessian=False, dgdf=None, d2gd2f=None):
        for i in range(len(self) - 1, -1, -1):
            dgdf, d2gd2f = self[i].backward(do_Hessian=do_Hessian, dgdf=dgdf, d2gd2f=d2gd2f)
        return dgdf, d2gd2f


class NNPytorchAD(nn.Module):

    def __init__(self, net: NN):
        super(NNPytorchAD, self).__init__()
        self.net = net
        self.ctx = None

    def forward(self, x, do_gradient=False, do_Hessian=False, **kwargs):
        (df, d2f) = (None, None)
        f, *_ = self.net(x)

        if do_gradient or do_Hessian:
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


class NNPytorchHessian(nn.Module):

    def __init__(self, net):
        super(NNPytorchHessian, self).__init__()
        self.net = net
        self.ctx = None

    def forward(self, x, do_gradient=False, do_Hessian=False, **kwargs):
        (df, d2f) = (None, None)
        f, *_ = self.net(x, do_gradient=False, do_Hessian=False)

        if f.squeeze().ndim > 1:
            raise ValueError(type(self), " must have scalar outputs per example")

        if do_gradient:
            df = grad(f.sum(), x)[0]
            df = df.unsqueeze(-1)

        if do_Hessian:
            d2f = hessian(lambda x: self.net(x)[0].sum(), x).sum(dim=2)
            d2f = d2f.unsqueeze(-1)

        return f, df, d2f


if __name__ == '__main__':
    import torch
    import hessQuik.activations as act
    import hessQuik.layers as lay
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    # problem setup
    nex = 11
    d = 3
    ms = [2, 7, 5]
    m = 1
    x = torch.randn(nex, d)

    f = NN(lay.singleLayer(d, ms[0], act=act.softplusActivation()),
           lay.singleLayer(ms[0], ms[1], act=act.softplusActivation()),
           lay.singleLayer(ms[1], ms[2], act=act.softplusActivation()),
           lay.singleLayer(ms[2], m, act=act.softplusActivation()))

    # f = NNPytorchHessian(f)
    # x.requires_grad = True

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=True)

    print('======= BACKWARD =======')
    input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
