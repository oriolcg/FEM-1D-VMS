import sys
import numpy as np


class Element(object):
    """1D element with basis functions of arbitrary order.

    Attributes:
        index (int): Index of the element.
        x_l (float): x-coordinate of the left boundary of the element.
        x_r (float): x-coordinate of the right boundary of the element.
    """

    def __init__(self, index, basis_function_order, x_l, x_r):
        self.local_nodes = 2
        self.index = index
        self.basis_function_order = basis_function_order
        self.x_l = x_l
        self.x_r = x_r
        self.xi_at_node = np.linspace(x_l, x_r, basis_function_order)

    def basis_function(self, x, local_node):
        x = np.asarray(x)
        y = np.zeros_like(x)

        if local_node == 0:
            y += ((x >= self.x_l) & (x <= self.x_r)) * \
                 (self.x_r - x) / (self.x_r - self.x_l)
        elif local_node == 1:
            y += ((x >= self.x_l) & (x <= self.x_r)) * \
                 (x - self.x_l) / (self.x_r - self.x_l)
        else:
            raise ValueError("Invalid local node")

        return y

    def test(self, node, xi):
        value = 1.0

        A = node

        for B in range(basis_function_order):
            if B != A:
                value *= (xi - xi_at_node(B)) / (xi_at_node(A) - xi_at_node(B))

        return value

    def basis_gradient(self, x, local_node):
        x = np.asarray(x)
        y = np.zeros_like(x)

        if local_node == 0:
            y += ((x >= self.x_l) & (x <= self.x_r)) * \
                 -1 / (self.x_r - self.x_l)
        elif local_node == 1:
            y += ((x >= self.x_l) & (x <= self.x_r)) * \
                 1 / (self.x_r - self.x_l)
        else:
            raise ValueError("Invalid local node")

        return y
