import sys
import numpy as np


class LinearElement(object):
    """1D element with linear basis functions.

    Attributes:
        index (int): Index of the element.
        x_l (float): x-coordinate of the left boundary of the element.
        x_r (float): x-coordinate of the right boundary of the element.
    """

    def __init__(self, index, x_l, x_r):
        self.local_nodes = 2
        self.index = index
        self.x_l = x_l
        self.x_r = x_r

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
