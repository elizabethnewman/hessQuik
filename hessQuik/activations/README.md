### Basics
Every hessQuik activation function has the following structure:
```python
sigma, dsigma, d2sigma = hessQuikActivationFunction(x, do_gradient=True, do_Hessian=True, forward_mode=True)
```

[comment]: <> (A key method of each layer is **compute_derivatives**, which is used for both forward and backward mode derivative computations.)

### For Contributors
New activation functions inherit the attributes and methods from ```hessQuikActivationFunction```.
The methods required to construct a new hessQuik activation layer are the following:
```python
def forward(self, *args, **kwargs):              # apply the activation function, call compute_derivatives if forward_mode = True
def compute_derivatives(self, *args, **kwargs):  # method to compute derivatives, used for forward and backward mode 
```
There is an additional method called ```backward``` that is automated when inheriting ```hessQuikActivationFunction```.

To test new activation functions, add the appropriate test to [tests/test_activations.py](https://github.com/elizabethnewman/hessQuik/tree/main/hessQuik/tests). 
