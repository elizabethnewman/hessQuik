import torch
import torch.nn as nn
import math
from hessQuik.layers import hessQuikLayer
import hessQuik.activations as act


class softmaxLayer(hessQuikLayer):
    """
    Forward propagation through single layer of the form

        f(u(x)) = act(u(x) @ K + b).

    Here, u(x) is the input into the layer and x is the input into the network of shapes

        x : (N, d) torch.Tensor
        u(x) : (N, in_features) torch.Tensor
        f(u(x)) : (N, out_features) torch.Tensor

    where N is the number of examples and d is the number of input features into the network.
    """

    def __init__(self, in_features):
        super(softmaxLayer, self).__init__()
        self.ctx = None
        self.in_features = in_features

    def dim_input(self):
        return self.in_features

    def dim_output(self):
        return self.in_features

    def forward(self, x, do_gradient=False, do_Hessian=False, forward_mode=True, dudx=None, d2ud2x=None):

        (dsigma, d2sigma) = (None, None)

        # forward propagate
        # sigma = torch.softmax(x, dim=1)
        sigma = x / x.sum(dim=1, keepdim=True)
        # sigma = x

        # compute derivatves
        if do_gradient or do_Hessian:
            if forward_mode is not None:
                dsigma, d2sigma = self.compute_derivatives(x, do_Hessian=do_Hessian)
            else:
                self.ctx = (x,)

        return sigma, dsigma, d2sigma

    def backward(self, *args, do_Hessian: bool = False):
        dsigma, d2sigma = self.compute_derivatives(*self.ctx, do_Hessian=do_Hessian)

        return dsigma, d2sigma

    def compute_derivatives(self, *args, do_Hessian=False):
        x = args[0]

        # exp_x = torch.exp(x)
        exp_x = x
        inv_sum_exp_x = 1.0 / exp_x.sum(dim=1, keepdim=True)
        dsigma = inv_sum_exp_x.unsqueeze(-1) * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        dsigma -= (inv_sum_exp_x ** 2).unsqueeze(-1) * (torch.ones_like(exp_x).unsqueeze(-1) * exp_x.unsqueeze(1))
        # dsigma = torch.ones_like(x).unsqueeze(-1) * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)

        d2sigma = None
        if do_Hessian:
            d2sigma = torch.zeros(x.shape[0], x.shape[1], x.shape[1], 1, device=x.device, dtype=x.dtype)

        return dsigma, d2sigma

    def extra_repr(self) -> str:
        return 'in_features={}'.format(
            self.in_features
        )


if __name__ == '__main__':
    from hessQuik.utils import input_derivative_check
    torch.set_default_dtype(torch.float64)

    nex = 11  # no. of examples
    d = 4  # no. of input features
    x = torch.randn(nex, d)

    f = softmaxLayer(d)

    print('======= FORWARD =======')
    input_derivative_check(f, x, do_Hessian=False, verbose=True, forward_mode=True)

    # print('======= BACKWARD =======')
    # input_derivative_check(f, x, do_Hessian=True, verbose=True, forward_mode=False)
