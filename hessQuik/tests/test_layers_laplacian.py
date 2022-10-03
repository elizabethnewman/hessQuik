import unittest
import torch
import hessQuik.activations as act
import hessQuik.layers as lay
from hessQuik.tests.utils import run_all_tests_laplacian


class TestLayer(unittest.TestCase):

    def test_singleLayer(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        d = 4  # no. of input features
        m = 7  # no. of output features
        x = torch.randn(nex, d)
        f = lay.singleLayer(d, m, act=act.softplusActivation())

        print(self)
        run_all_tests_laplacian(f, x)

    def test_resnetLayer(self):
        torch.set_default_dtype(torch.float64)
        nex = 11  # no. of examples
        width = 4  # no. of input features
        h = 0.25
        x = torch.randn(nex, width)
        f = lay.resnetLayer(width, h=h, act=act.softplusActivation())
        print(self)
        run_all_tests_laplacian(f, x)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
