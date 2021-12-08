import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
from hessQuik.tests.utils import DerivativeCheckTestsNetwork


class TestLayer(unittest.TestCase):

    @staticmethod
    def run_test(f, x, dx):
        derivativeTests = DerivativeCheckTestsNetwork()

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

        print(self)
        self.run_test(f, x, dx)

    def test_resnetLayer(self):
        nex = 11  # no. of examples
        width = 4  # no. of input features
        h = 0.25
        x = torch.randn(nex, width)
        dx = torch.randn_like(x)
        f = lay.resnetLayer(width, h=h, act=act.softplusActivation())
        print(self)
        self.run_test(f, x, dx)

    def test_ICNNLayer(self):
        # problem setup
        nex = 11  # no. of examples
        d = 3  # no. of input features
        m = 5  # no. of output features
        x = torch.randn(nex, d)
        dx = torch.randn_like(x)
        f = lay.ICNNLayer(d, None, m, act=act.softplusActivation())

        print(self)
        self.run_test(f, x, dx)

    def test_quadraticLayer(self):
        # problem setup
        nex = 11  # no. of examples
        d = 4  # no. of input dimensiona features
        m = 7  # rank
        x = torch.randn(nex, d)
        dx = torch.randn_like(x)
        f = lay.quadraticLayer(d, m)

        print(self)
        self.run_test(f, x, dx)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
