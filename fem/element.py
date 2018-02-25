import sys
import numpy as np


class LinearElement(object):
    """1D element with linear basis functions.

    Attributes:
        index_l (int): Index of the left node of the element.
        index_r (int): Index of the right node of the element.
        x_l (float): x-coordinate of the left boundary of the element.
        x_r (float): x-coordinate of the right boundary of the element.
    """

    def __init__(self, index, x_l, x_r):
        self.index = index
        self.x_l = x_l
        self.x_r = x_r

    def W_l(self, x):
        return (self.x_r - x) / (self.x_r - self.x_l)

    def dW_l(self, x):
        return -1 / (self.x_r - self.x_l)

    def W_r(self, x):
        return (x - self.x_l) / (self.x_r - self.x_l)

    def dW_r(self, x):
        return 1 / (self.x_r - self.x_l)

    def factor_u_l(self, x):
        return (self.x_r - x) / (self.x_r - self.x_l)

    def factor_du_l(self, x):
        return -1 / (self.x_r - self.x_l)

    def factor_u_r(self, x):
        return (x - self.x_l) / (self.x_r - self.x_l)

    def factor_du_r(self, x):
        return 1 / (self.x_r - self.x_l)
