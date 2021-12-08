import unittest
import torch
import activations as act
import layers as lay
from test_utils import DerivativeCheckTests


class DerivativeCheckTestsLayer(DerivativeCheckTests):
    def get_directional_derivatives(self, dx, df0, d2f0=None):
        dfdx = torch.matmul(df0.transpose(1, 2), dx.unsqueeze(2)).squeeze(2)
        if d2f0 is None:
            return dfdx
        else:
            curvx = torch.sum(dx.unsqueeze(2).unsqueeze(3) * d2f0 * dx.unsqueeze(1).unsqueeze(3), dim=(1, 2))
            return dfdx, curvx


class TestLayer(unittest.TestCase):

    @staticmethod
    def run_test(f, x, dx):
        derivativeTests = DerivativeCheckTestsLayer()

        # forward tests
        derivativeTests.run_forward_gradient_test(f, x, dx)
        derivativeTests.run_forward_hessian_test(f, x, dx)
        derivativeTests.run_backward_gradient_test(f, x, dx)
        derivativeTests.run_backward_hessian_test(f, x, dx)

    def test_singleLayer(self):
        # problem setup
        nex = 11  # no. of examples
        d = 4  # no. of input features
        m = 7  # no. of output features
        x = torch.randn(nex, d)
        dx = torch.randn_like(x)
        f = lay.singleLayer(d, m, act=act.softplusActivation())

        print(type(f))
        self.run_test(f, x, dx)

    def test_resnetLayer(self):
        nex = 11  # no. of examples
        width = 4  # no. of input features
        h = 0.25
        x = torch.randn(nex, width)
        dx = torch.randn_like(x)
        f = lay.resnetLayer(width, h=h, act=act.softplusActivation())
        print(type(f))
        self.run_test(f, x, dx)

    # def test_ICNNLayer(self):
    #     nex = 11  # no. of examples
    #     input_dim = 4
    #     in_features = 3
    #     out_features = 5
    #
    #     x = torch.randn(nex, input_dim)
    #     u = torch.zeros(nex, in_features)
    #     dx = torch.randn_like(x)
    #     du = torch.zeros_like(u)
    #
    #     f = lay.ICNNLayer2(input_dim, in_features, out_features, act=act.softplusActivation())
    #     print(type(f))
    #     self.run_test(f, torch.cat((u, x), dim=1), torch.cat((du, dx), dim=1))


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()