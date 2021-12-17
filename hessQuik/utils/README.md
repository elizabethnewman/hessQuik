## Options

We offer two types of derivative tests for the input of the network.
```python
def input_derivative_check(*args, **kwargs):
def input_derivative_check_finite_difference(*args, **kwargs):
```
The first uses a Taylor approximation to determine the accuracy of the derivatives.  

### Taylor Series Test 
For example,
If our gradient is correct, then if we perturb our input $\mathbf{x}$ by unit vector $\mathbf{p}$ with step size $h$, then by Taylor's Theorem,
````latex
\left\|f(\mathbf{x}) + h \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}^\top \mathbf{p} - f(\mathbf{x} + h\mathbf{p})\right\| = \mathcal{O}(h^2).
````
If we have computed the correct gradient, $\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}$, then as $h\to 0$, the first-order error (left-hand side) also goes to zero at a rate of $h^2$.  This means if we divide $h$ by $2$ (smaller perturbation), then we should see the first-order error decrease by a factor of $4$. Similar logic can be applied for the zeroth-order (no derivative information) and second-order (Hessian information) error.