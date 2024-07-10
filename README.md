# fem-challenge-JT
FEM challenge from Jeremy Theler @ [Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7216466748546400256/)

# My Solution

### Problem Statement

First we consider the curved second-order `line3` element lying on the x-y plane defined by the corner nodes:

$$
\mathbf{x}_1 = [0, 1]
$$

$$
\mathbf{x}_2 = [1, 0]
$$

with the mid-edge node located at:

$$
\mathbf{x}_3 = \left[ \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right]
$$

Using the shape functions for $\xi \in [-1, 1]$:

$$
h_1(\xi) = \frac{\xi \cdot (\xi - 1)}{2}
$$

$$
h_2(\xi) = \frac{\xi \cdot (\xi + 1)}{2}
$$

$$
h_3(\xi) = (1 + \xi) \cdot (1 - \xi)
$$

compute the Lebesgue measure (i.e., length) of the element. The real length of the curve is $\pi/2 \approx 1.5708$.

## Mathematical Description

The position $\mathbf{x}(\xi)$ on the element can be described as:

$$
\mathbf{x}(\xi) = h_1(\xi) \cdot \mathbf{x}_1 + h_2(\xi) \cdot \mathbf{x}_2 + h_3(\xi) \cdot \mathbf{x}_3
$$

To find the length of the element, we need to integrate the arc length differential:

$$
L = \int_{-1}^{1} \left\| \frac{d\mathbf{x}(\xi)}{d\xi} \right\| d\xi
$$

First, we find the derivative of $\mathbf{x}(\xi)$:

$$
\frac{d\mathbf{x}(\xi)}{d\xi} = \frac{d}{d\xi} \left( h_1(\xi) \cdot \mathbf{x}_1 + h_2(\xi) \cdot \mathbf{x}_2 + h_3(\xi) \cdot \mathbf{x}_3 \right)
$$

We can compute the derivatives of the shape functions:

$$
\frac{dh_1(\xi)}{d\xi} = \xi - \frac{1}{2}
$$

$$
\frac{dh_2(\xi)}{d\xi} = \xi + \frac{1}{2}
$$

$$
\frac{dh_3(\xi)}{d\xi} = -2\xi
$$

Therefore:

$$
\frac{d\mathbf{x}(\xi)}{d\xi} = \left( \xi - \frac{1}{2} \right) \cdot \mathbf{x}_1 + \left( \xi + \frac{1}{2} \right) \cdot \mathbf{x}_2 - 2\xi \cdot \mathbf{x}_3
$$

Substitute the node coordinates:

$$
\frac{d\mathbf{x}(\xi)}{d\xi} = \left( \xi - \frac{1}{2} \right) \cdot [0, 1] + \left( \xi + \frac{1}{2} \right) \cdot [1, 0] - 2\xi \cdot \left[ \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right]
$$

Simplifying:

$$
\frac{d\mathbf{x}(\xi)}{d\xi} = \left[ \left( \xi + \frac{1}{2} \right) - \frac{2\xi}{\sqrt{2}}, \left( \xi - \frac{1}{2} \right) - \frac{2\xi}{\sqrt{2}} \right]
$$

$$
\frac{d\mathbf{x}(\xi)}{d\xi} = \left[ \xi + \frac{1}{2} - \frac{2\xi}{\sqrt{2}}, \xi - \frac{1}{2} - \frac{2\xi}{\sqrt{2}} \right]
$$

Next, we compute the magnitude of this derivative:

$$
\left\| \frac{d\mathbf{x}(\xi)}{d\xi} \right\| = \sqrt{ \left( \xi + \frac{1}{2} - \frac{2\xi}{\sqrt{2}} \right)^2 + \left( \xi - \frac{1}{\sqrt{2}} - \frac{2\xi}{\sqrt{2}} \right)^2 }
$$

Finally, integrate this magnitude over $[-1, 1]$ to find the length $L$:

$$
L = \int_{-1}^{1} \sqrt{ \left( \xi + \frac{1}{2} - \frac{2\xi}{\sqrt{2}} \right)^2 + \left( \xi - \frac{1}{\sqrt{2}} - \frac{2\xi}{\sqrt{2}} \right)^2 } d\xi
$$

## Python Solution

We can do multiple approaches to this, first would be using `quad` integration from `scipy`:

```python
import numpy as np
from scipy.integrate import quad

# Node coordinates
x1 = np.array([0, 1])
x2 = np.array([1, 0])
x3 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# Shape function derivatives
def dh1_dxi(xi):
    return xi - 0.5

def dh2_dxi(xi):
    return xi + 0.5

def dh3_dxi(xi):
    return -2 * xi

# Derivative of x with respect to xi
def dx_dxi(xi):
    dx = dh1_dxi(xi) * x1 + dh2_dxi(xi) * x2 + dh3_dxi(xi) * x3
    return dx

# Magnitude of the derivative
def integrand(xi):
    dx = dx_dxi(xi)
    return np.sqrt(np.sum(dx**2))

# Compute the length by numerical integration
length, error = quad(integrand, -1, 1)

print(f"Computed length of the element: {length:.4f}")
```

But this would be too costly, so we can use just use `numpy` and define the quadrature ourselves

```python
# Gaussian quadrature points and weights for three-point rule
gauss_points = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
gauss_weights = [5/9, 8/9, 5/9]

# Compute the length by Gaussian quadrature
length = 0
for xi, w in zip(gauss_points, gauss_weights):
    length += w * integrand(xi)

print(f"Computed length of the element using Gaussian quadrature: {length:.4f}")
```

If we want to visualize what we are integrating (i.e., shape function) and the actual quarter circle

```python
import matplotlib.pyplot as plt

# Position function x(xi) based on the shape functions
def x_xi(xi):
    h1 = 0.5 * xi * (xi - 1)
    h2 = 0.5 * xi * (xi + 1)
    h3 = 1 - xi**2
    return h1 * x1 + h2 * x2 + h3 * x3

# Generate xi values and corresponding x(xi) values
xi_vals = np.linspace(-1, 1, 100)
x_vals = np.array([x_xi(xi) for xi in xi_vals])

# Generate points for the quarter circle
theta = np.linspace(0, np.pi/2, 100)
quarter_circle_x = np.cos(theta)
quarter_circle_y = np.sin(theta)

# Plot the curve defined by x(ξ) and the quarter circle
plt.plot(x_vals[:, 0], x_vals[:, 1], label='x(ξ)', linestyle='--')
plt.plot(quarter_circle_x, quarter_circle_y, label='Quarter Circle', linestyle=':')
plt.scatter([x1[0], x2[0], x3[0]], [x1[1], x2[1], x3[1]], color='red', label='Nodes')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Defined by x(ξ) and Quarter Circle')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

![Image](/plot.png)
