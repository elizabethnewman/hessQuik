### Basics

Each hessQuik layer has the following structure:
```python
f, df, d2f = hessQuikLayer(x, do_gradient=True, do_Hessian=True, forward_mode=True, dudx=None, d2ud2x=None)
```
The ```forward_mode``` flag indicates if the derivatives should be computed during the forward pass through the network or in backward mode after propagating through the network. 
The role of the ```forward_mode``` flag is the following:
```python
forward_mode = True  # compute derivatives in forward mode
forward_mode = None  # store info to compute derivatives in backward mode, but wait to compute
forward_mode = False # compute the derivatives in backward mode immediately
```
No matter the setting of ```forward_mode```, the output of ```hessQuikLayer``` will include the gradients and Hessians if the appropriate flags are turned on.

### For Contributors
All hessQuik layers should inherit the methods from ```hessQuikLayer``` which include
```python
def dim_input(self):               # return the dimension of the input into the layer 
def dim_output(self):              # return the dimension of the output from the layer
def forward(self, *args, **kwargs):  # propagate through network and compute derivatives in either forward or backward more
def backward(self, **kwargs):        # method to compute derivatives in backward mode; this method will be called in forward()
```
To test new layers, add the appropriate test to [tests/test_layers.py](https://github.com/elizabethnewman/hessQuik/tree/main/hessQuik/tests). 




