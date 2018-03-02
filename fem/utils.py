import sys
import math
import numpy as np
import matplotlib.pyplot as plt


class Utils(object):

    @staticmethod
    def integrate(functions, x_l, x_r, N_q):
        """Performs guassian quadrature on a function or product of multiple
        functions.

        Arguments:
            functions: A function or constant OR a tuple of functions and
                constants. If the argument is a tuple, then the integrand is
                product of the functions and constants in the tuple.
            x_l: Leftmost x-coordinate of the integration interval.
            x_r: Rightmost x-coordinate of the integration interval.
            N_q: Number of Guassian quadrature points up to a maximum of 3.

        Returns:
            The integral.
        """

        # Make sure the "functions" variable is a tuple even if a single
        # function was supplied as input.
        if not isinstance(functions, tuple):
            functions = (functions,)

        # Set quadrature rule. "w_q" is a list of weights and "xi_q" is a list
        # of quadrature points in the domain -1 < xi < 1.
        if N_q == 1:
            w_q = [2.0]
            xi_q = [0.0]
        elif N_q == 2:
            w_q = [1.0, 1.0]
            xi_q = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
        elif N_q == 3:
            w_q = [5/9, 8/9, 5/9]
            xi_q = [-1 * math.sqrt(3/5), 0, math.sqrt(3/5)]
        else:
            raise ValueError("Unknown quadrature rule.")

        result = 0

        # Loop over the indices of the quadrature points
        for nn in range(N_q):
            # Get xi location of quadrature point
            xi = xi_q[nn]

            # Calculate x location of quadrature point
            x_q = x_l + 0.5 * (1 + xi) * (x_r - x_l)

            # Compute the value of the integrand at the quadrature point
            f = 1

            for function in functions:
                if callable(function):
                    f *= function(x_q)
                else:
                    f *= function

            # Add contribution to result
            result += w_q[nn] * 0.5 * (x_r - x_l) * f

        return result
