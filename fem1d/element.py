import sys
import numpy as np


class LinearElement(object):
    """1D element with linear basis functions.

    Attributes:
        index (int): Index of the element.
        x_l (float): x-coordinate of the left boundary of the element.
        x_r (float): x-coordinate of the right boundary of the element.
    """

    def __init__(self, index, x_left, x_right):
        self.num_nodes = 2
        self.index = index
        self.x_left = x_left
        self.x_right = x_right
        self.h = x_right - x_left

    def basis_function(self, x, local_node):
        #
        # Inputs:
        #
        #   (int OR numpy.ndarray) local_node, the index of the basis
        #       function.
        #
        #   (float) x, the evaluation point.
        #
        # Outputs:
        #
        #   (float) phi, the value of the basis function at x.
        #

        x = np.asarray(x)
        phi = np.zeros_like(x)

        if local_node == 0:
            phi += ((x >= self.x_left) & (x <= self.x_right)) * \
                   (self.x_right - x) / (self.x_right - self.x_left)
        elif local_node == 1:
            phi += ((x >= self.x_left) & (x <= self.x_right)) * \
                   (x - self.x_left) / (self.x_right - self.x_left)
        else:
            raise ValueError("Invalid local node")

        return phi

    def basis_gradient(self, x, local_node):
        #
        # Inputs:
        #
        #   (int OR numpy.ndarray) local_node, the index of the basis
        #       function.
        #
        #   (float) x, the evaluation point.
        #
        # Outputs:
        #
        #   (float) phi_x, the value of the derivative of the basis
        #       function at x.
        #

        x = np.asarray(x)
        phi_x = np.zeros_like(x)

        if local_node == 0:
            phi_x += ((x >= self.x_left) & (x <= self.x_right)) * \
                     -1 / (self.x_right - self.x_left)
        elif local_node == 1:
            phi_x += ((x >= self.x_left) & (x <= self.x_right)) * \
                     1 / (self.x_right - self.x_left)
        else:
            raise ValueError("Invalid local node")

        return phi_x
