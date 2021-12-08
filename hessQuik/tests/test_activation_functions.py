import unittest
import torch
import hessQuik.layers.activation_functions as act
from hessQuik.tests.utils import DerivativeCheckTestsActivationFunction


class TestActivation(unittest.TestCase):

    @staticmethod
    def run_test(f):
        # problem setup
        nex = 11  # no. of examples
        d = 4  # no. of input features

        x = torch.randn(nex, d)
        dx = torch.randn_like(x)

        derivativeTests = DerivativeCheckTestsActivationFunction()

        # forward tests
        derivativeTests.run_forward_gradient_test(f, x, dx)
        derivativeTests.run_forward_hessian_test(f, x, dx)
        derivativeTests.run_backward_gradient_test(f, x, dx)
        derivativeTests.run_backward_hessian_test(f, x, dx)

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


def main():
    torch.set_default_dtype(torch.float64)
    unittest.main()


if __name__ == '__main__':
    main()
