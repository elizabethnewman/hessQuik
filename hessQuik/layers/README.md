### hessQuik Layers

The fundamental layer for the hessQuik package is found in **hessQuik_layer.py**.  The layer has the following structure:
```python
f, df, d2f = hessQuikLayer(x, do_gradient=True, do_Hessian=True)
```
A key attribute of each layer is **reverse_mode**, which is defined as follows:
```python
reverse_mode = False  # compute derivatives in forward mode
reverse_mode = None   # store necessary information to compute derivatives in backward mode
reverse_mode = True   # compute derivatives in backward mode immediately  
```


