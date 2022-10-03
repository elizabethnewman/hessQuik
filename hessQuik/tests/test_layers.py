import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
from hessQuik.tests.utils import run_all_tests


class TestLayer(unittest.TestCase):

    def test_singleLayer(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        d = 4  # no. of input features
        m = 7  # no. of output features
        x = torch.randn(nex, d)
        f = lay.singleLayer(d, m, act=act.softplusActivation())

        print(self)
        run_all_tests(f, x)

    def test_singleLayer_no_bias(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        d = 4  # no. of input features
        m = 7  # no. of output features
        x = torch.randn(nex, d)
        f = lay.singleLayer(d, m, act=act.softplusActivation(), bias=False)

        print(self)
        run_all_tests(f, x)

    def test_resnetLayer(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        width = 4  # no. of input features
        h = 0.25
        x = torch.randn(nex, width)
        f = lay.resnetLayer(width, h=h, act=act.softplusActivation())
        print(self)
        run_all_tests(f, x)

    def test_resnetLayer_no_bias(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        width = 4  # no. of input features
        h = 0.25
        x = torch.randn(nex, width)
        f = lay.resnetLayer(width, h=h, act=act.softplusActivation(), bias=False)
        print(self)
        run_all_tests(f, x)

    def test_ICNNLayer(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        d = 3  # no. of input features
        m = 5  # no. of output features
        x = torch.randn(nex, d)
        f = lay.ICNNLayer(d, None, m, act=act.softplusActivation())

        print(self)
        run_all_tests(f, x)

    def test_quadraticLayer(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        d = 4  # no. of input dimensiona features
        m = 7  # rank
        x = torch.randn(nex, d)
        f = lay.quadraticLayer(d, m)

        print(self)
        run_all_tests(f, x)

    def test_quadraticICNNLayer(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        d = 3  # no. of input features
        m = 5  # no. of output features

        x = torch.randn(nex, d)
        f = lay.quadraticICNNLayer(d, None, m)

        print(self)
        run_all_tests(f, x)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
