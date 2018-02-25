import sys
import numpy as np
from fem.weakformterm import WeakFormTerm


class WeakForm(object):
    """Weak form of the PDE.

    Attributes:
        terms (list of dictionaries): List of dictionaries describing the terms
            in the weak form of the PDE.
        source_function (function): Reference to the source function.
        bc_l (BoundaryCondition): Left boundary condition object.
        bc_r (BoundaryCondition): Right boundary condition object.
    """

    def __init__(self):
        self.terms = []
        self.source_function = False
        self.bc_l = None
        self.bc_r = None

    def add_w_u(self, constant=1):
        """
        Add a term of the form
            \int c w(x) u(x) dx
        to the weak form.

        Args:
            constant: Multiplication constant.
        """

        self.terms.append(WeakFormTerm.w_u(constant))

    def add_w_du(self, constant=1):
        """
        Add a term of the form
            \int c w(x) du/dx dx
        to the weak form.

        Args:
            constant: Multiplication constant.
        """

        self.terms.append(WeakFormTerm.w_du(constant))

    def add_dw_u(self, constant=1):
        """
        Add a term of the form
            \int c dw/dx u(x) dx
        to the weak form.

        Args:
            constant: Multiplication constant.
        """

        self.terms.append(WeakFormTerm.dw_u(constant))

    def add_dw_du(self, constant=1):
        """
        Add a term of the form
            \int c dw/dx du/dx dx
        to the weak form.

        Args:
            constant: Multiplication constant.
        """

        self.terms.append(WeakFormTerm.dw_du(constant))
