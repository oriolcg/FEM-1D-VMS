import math
import numpy as np

class QuadratureRule(object):

    def __init__(self, num_quad_points):
        self.num_quad_points = num_quad_points
        self.xi_q, self.w_q = np.polynomial.legendre.leggauss(num_quad_points)
