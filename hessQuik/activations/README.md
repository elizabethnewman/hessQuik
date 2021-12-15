### Basics
Every hessQuik activation function has the following structure:
```python
sigma, dsigma, d2sigma = hessQuikActivationFunction(x, do_gradient=True, do_Hessian=True, forward_mode=True)
```

[comment]: <> (A key method of each layer is **compute_derivatives**, which is used for both forward and backward mode derivative computations.)

### For Developers
New activation functions inherit the attributes and methods from ```hessQuikActivationFunction```.
The methods required to construct a new hessQuik activation layer are the following:
```html
forward             : apply the activation function, call compute_derivatives if forward_mode = True
compute_derivatives : method to compute derivatives, used for forward and backward mode 
```
There is an additional method called ```backward``` that is automated in superclass.