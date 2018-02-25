import sys
import numpy as np


class BoundaryCondition(object):
    """Boundary condition."""

    def __init__(self, boundary_condition_type, value):
        self.type = boundary_condition_type
        self.value = value

    @classmethod
    def dirichlet(cls, value):
        return cls("Dirichlet", value)

    @classmethod
    def neumann_left(cls, value):
        return cls("Neumann", value)

    @classmethod
    def robin_left(cls, value):
        return cls("Robin", value)
