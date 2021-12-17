## Options

We offer a derivative test for the input of the network and a derivative test for the weights/parameters of the network.
```python
def input_derivative_check(f, x, do_Hessian=False, forward_mode=True, **kwargs):
def network_derivative_check(f, x, do_Hessian=False, forward_mode=True, **kwargs)
```
The first uses a Taylor approximation to determine the accuracy of the derivatives.  This is the test we use primarily and that we recommend.  We provide some intuition for this test below.

We also provide a finite difference check for the derivatives of the network with respect to the inputs.  This test tends to be slower than the Taylor series test, but can be useful for debugging. 

### Taylor Approximation Test 
For example, suppose we compute the value and derivatives for a given input ```x``` via
```python
f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)
```
Suppose we perturb the input using a normalized tensor ```p``` and a step size ```h``` via
```python
fh, *_ = f(x + h * p, do_gradient=False, do_Hessian=False)
```
If the derivatives are computed correctly, the Taylor approximation test will observe the following behavior. 
Let ```h``` approach 0. Then, we should observe that 
* The zeroth-order Taylor approximation to```fh``` about ```x``` approaches ```f0``` at the rate that ```h``` approaches 0.
* The first-order Taylor approximation to ```fh``` about ```x``` approaches ```f0``` at the rate that ```h^2``` approaches 0.
* The second-order Taylor approximation to ```fh``` about ```x``` approaches ```f0``` at the rate that ```h^3``` approaches 0.

