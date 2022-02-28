import unittest
import torch
import hessQuik.activations as act
from hessQuik.tests.utils import run_all_tests


class TestActivation(unittest.TestCase):

    @staticmethod
    def run_test(f):
        torch.set_default_dtype(torch.float64)
        nex = 11
        d = 7
        x = torch.randn(nex, d)
        run_all_tests(f, x)

    def test_antiTanhActivation(self):
        torch.set_default_dtype(torch.float64)
        f = act.antiTanhActivation()
        print(type(f))
        self.run_test(f)

    def test_quadraticActivation(self):
        torch.set_default_dtype(torch.float64)
        f = act.quadraticActivation()
        print(type(f))
        self.run_test(f)

    def test_softplusActivation(self):
        torch.set_default_dtype(torch.float64)
        f = act.softplusActivation()
        print(type(f))
        self.run_test(f)

    def test_identityActivation(self):
        torch.set_default_dtype(torch.float64)
        f = act.identityActivation()
        print(type(f))
        self.run_test(f)

    def test_tanhActivation(self):
        torch.set_default_dtype(torch.float64)
        f = act.tanhActivation()
        print(type(f))
        self.run_test(f)

    def test_sigmoidActivation(self):
        torch.set_default_dtype(torch.float64)
        f = act.sigmoidActivation()
        print(type(f))
        self.run_test(f)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
