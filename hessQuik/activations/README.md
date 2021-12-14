The fundamental activation function for the hessQuik package is found in **activation_function.py**.  The layer has the following structure:
```python
sigma, dsigma, d2sigma = hessQuikActivationFunction(x, do_gradient=True, do_Hessian=True)
```
A key method of each layer is **compute_derivatives**, which is used for both forward and backward mode derivative computations.

