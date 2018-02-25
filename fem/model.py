import sys
import numpy as np
import math
from fem.utils import Utils


class Model(object):
    """FEM model."""

    def __init__(self, mesh, weak_form, integration_points=3):
        self.mesh = mesh
        self.weak_form = weak_form
        self.N_q = integration_points
        self.K = None
        self.u = None
        self.F = None

    def solve(self):
        """Solve the system of linear equations."""

        self.__assemble()
        self.u = np.linalg.solve(self.K, self.F)

    def __assemble(self):
        """Assemble the stiffness matrix K and the right-hand side vector F."""

        self.K = np.zeros((self.mesh.num_elements+1, self.mesh.num_elements+1))
        self.F = np.zeros(self.mesh.num_elements+1)

        for element in self.mesh.elements:
            # The variables l and r are indices of respectively the leftmost
            # and rightmost nodes of the linear element
            l = element.index
            r = element.index + 1

            # The variables x_l and x_r are respectively the x-coordinates of
            # the left and right boundaries of the linear element
            x_l = element.x_l
            x_r = element.x_r

            # Assemble the stiffness matrix K
            for term in self.weak_form.terms:
                if term.type == "dW_du":
                    # Left basis function and left node
                    self.K[l][l] += Utils.integrate((term.constant,
                                                     element.dW_l,
                                                     element.factor_du_l),
                                                    x_l, x_r, self.N_q)

                    # Left basis function and right node
                    self.K[l][r] += Utils.integrate((term.constant,
                                                     element.dW_l,
                                                     element.factor_du_r),
                                                    x_l, x_r, self.N_q)

                    # Right basis function and left node
                    self.K[r][l] += Utils.integrate((term.constant,
                                                     element.dW_r,
                                                     element.factor_du_l),
                                                    x_l, x_r, self.N_q)

                    # Right basis function and right node
                    self.K[r][r] += Utils.integrate((term.constant,
                                                     element.dW_r,
                                                     element.factor_du_r),
                                                    x_l, x_r, self.N_q)
                elif term.type == "W_du":
                    self.K[l][l] += Utils.integrate((term.constant,
                                                     element.W_l,
                                                     element.factor_du_l),
                                                    x_l, x_r, self.N_q)
                    self.K[l][r] += Utils.integrate((term.constant,
                                                     element.W_l,
                                                     element.factor_du_r),
                                                    x_l, x_r, self.N_q)
                    self.K[r][l] += Utils.integrate((term.constant,
                                                     element.W_r,
                                                     element.factor_du_l),
                                                    x_l, x_r, self.N_q)
                    self.K[r][r] += Utils.integrate((term.constant,
                                                     element.W_r,
                                                     element.factor_du_r),
                                                    x_l, x_r, self.N_q)
                elif term.type == "dW_u":
                    self.K[l][l] += Utils.integrate((term.constant,
                                                     element.dW_l,
                                                     element.factor_u_l),
                                                    x_l, x_r, self.N_q)
                    self.K[l][r] += Utils.integrate((term.constant,
                                                     element.dW_l,
                                                     element.factor_u_r),
                                                    x_l, x_r, self.N_q)
                    self.K[r][l] += Utils.integrate((term.constant,
                                                     element.dW_r,
                                                     element.factor_u_l),
                                                    x_l, x_r, self.N_q)
                    self.K[r][r] += Utils.integrate((term.constant,
                                                     element.dW_r,
                                                     element.factor_u_r),
                                                    x_l, x_r, self.N_q)
                else:  # term.type == "W_u"
                    self.K[l][l] += Utils.integrate((term.constant,
                                                     element.W_l,
                                                     element.factor_u_l),
                                                    x_l, x_r, self.N_q)
                    self.K[l][r] += Utils.integrate((term.constant,
                                                     element.W_l,
                                                     element.factor_u_r),
                                                    x_l, x_r, self.N_q)
                    self.K[r][l] += Utils.integrate((term.constant,
                                                     element.W_r,
                                                     element.factor_u_l),
                                                    x_l, x_r, self.N_q)
                    self.K[r][r] += Utils.integrate((term.constant,
                                                     element.W_r,
                                                     element.factor_u_r),
                                                    x_l, x_r, self.N_q)

            # Assemble the right-hand side vector F
            if self.weak_form.source_function:
                # The term on the right-hand side is always of the form
                # \int W(x) f(x) dx
                self.F[l] += Utils.integrate((element.W_l,
                                              self.weak_form.source_function),
                                             x_l, x_r, self.N_q)
                self.F[r] += Utils.integrate((element.W_r,
                                              self.weak_form.source_function),
                                             x_l, x_r, self.N_q)

        # Set left boundary condition
        if self.weak_form.bc_l.type == "Dirichlet":
            self.F[0] = self.weak_form.bc_l.value
            self.K[0, :] = 0
            self.K[0, 0] = 1
        else:  # bc.type == "Neumann"
            self.F[0] += -1 * self.weak_form.bc_l.value

        # Set right boundary condition
        if self.weak_form.bc_r.type == "Dirichlet":
            self.F[len(self.F)-1] = self.weak_form.bc_r.value
            self.K[len(self.F)-1, :] = 0
            self.K[len(self.F)-1, len(self.F)-1] = 1
        else:  # bc.type == "Neumann"
            self.F[len(self.F)-1] += self.weak_form.bc_r.value
