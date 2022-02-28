import unittest
import torch
import hessQuik.activations as act
from utils import run_all_tests


class TestActivation(unittest.TestCase):

    @staticmethod
    def run_test(f):
        nex = 11
        d = 7
        x = torch.randn(nex, d)
        run_all_tests(f, x)

    def test_antiTanhActivation(self):
        f = act.antiTanhActivation()
        print(type(f))
        self.run_test(f)

    def test_quadraticActivation(self):
        f = act.quadraticActivation()
        print(type(f))
        self.run_test(f)

    def test_softplusActivation(self):
        f = act.softplusActivation()
        print(type(f))
        self.run_test(f)

    def test_identityActivation(self):
        f = act.identityActivation()
        print(type(f))
        self.run_test(f)

    def test_tanhActivation(self):
        f = act.tanhActivation()
        print(type(f))
        self.run_test(f)

    def test_sigmoidActivation(self):
        f = act.sigmoidActivation()
        print(type(f))
        self.run_test(f)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    unittest.main()
