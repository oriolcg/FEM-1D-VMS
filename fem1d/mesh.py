import sys
import math
import numpy as np
from fem1d.element import LinearElement


class Mesh(object):
    def __init__(self, x, elements):
        self.x = x
        self.elements = elements
        self.num_elements = len(elements)

    @classmethod
    def uniform_grid(cls, x_start, x_end, num_elements):
        """
        Creates a 1D uniform grid.

        Args:
            x_start: Leftmost x-coordinate of the domain.
            x_end: Rightmost x-coordinate of the domain.

        Returns:
            A fully initialized instance of Mesh.
        """

        x = np.linspace(x_start, x_end, num_elements+1)

        # Create list of instances of Element
        elements = [LinearElement(i, x[i], x[i+1]) for i in range(len(x)-1)]

        return cls(x, elements)

    @classmethod
    def non_uniform_grid(cls, x_start, x_end, num_elements, ratio):
        """
        Creates a 1D non-uniform grid by placing num_elements nodes in
        geometric progression between x_start and x_end with ratio ratio.

        Args:
            x_start: This is the first param.
            x_end: This is a second param.

        Returns:
            A fully initialized instance of Mesh.
        """

        # Create grid points between 0 and 1
        h = (ratio-1) / (ratio**num_elements - 1)
        x = np.append([0], h * np.cumsum(ratio**np.arange(num_elements)))

        # Scale to start at x_start and end at x_end
        x = x_start + x * (x_end-x_start)

        # Create list of instances of Element
        elements = [LinearElement(i, x[i], x[i+1]) for i in range(len(x)-1)]

        return cls(x, elements)
