
### Basics
We provide unit testing of the gradients and Hessians of hessQuik activation functions, layers, and networks.  The unit tests use Python's [unittest](https://docs.python.org/3/library/unittest.html) framework. The main test files are the following:
```html
test_activations.py
test_layers.py
test_networks.py
test_network_weight_derivatives.py
```
The first three test the gradients and Hessians with respect to the inputs.  The last test checks the gradients of the networks with respect to the network weights computed using PyTorch's automatic differentiation.

We provide two functions to check the derivatives for the inputs and weights of the network.  See [utils](https://github.com/elizabethnewman/hessQuik/tree/main/hessQuik/utils) for details.

### For Contributors

To run all tests, use the following from the command line
```console
python test_activations.py
```

To run one specific test, use the following from the command line
```console
python test_layers.py TestLayer.test_singleLayer
```

When writing new derivative tests, be sure to check four cases:
```python
(1) do_Hessian = False,  forward_mode = True
(2) do_Hessian = True,   forward_mode = True
(3) do_Hessian = False,  forward_mode = False
(4) do_Hessian = True,   forward_mode = False
```

