import sys
import numpy as np
import math
from functools import partial
from fem.utils import Utils


class Model(object):
    """FEM model."""

    def __init__(self, mesh, weak_form, quad_points=3):
        self.mesh = mesh
        self.weak_form = weak_form
        self.quad_points = quad_points
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

        # Loop over the elements in the mesh
        for element in self.mesh.elements:
            # The variables x_l and x_r are respectively the x-coordinates of
            # the left and right boundaries of the element
            x_l = element.x_l
            x_r = element.x_r

            # Assemble the stiffness matrix K by looping over the integrals in
            # the weak form
            for integral in self.weak_form.terms:
                # Select the functions in the integrand
                if integral.type == "dW_du":
                    W_func = element.basis_gradient
                    u_func = element.basis_gradient
                elif integral.type == "W_du":
                    W_func = element.basis_function
                    u_func = element.basis_gradient
                elif integral.type == "dW_u":
                    W_func = element.basis_gradient
                    u_func = element.basis_function
                elif integral.type == "W_u":
                    W_func = element.basis_function
                    u_func = element.basis_function
                else:
                    raise ValueError("Unknown type of WeakFormTerm")

                # Loop over the local nodes in the element
                for i in range(element.local_nodes):
                    for j in range(element.local_nodes):
                        # Create a tuple of constants and/or callables
                        functions = (integral.constant,
                                     partial(W_func, local_node=i),
                                     partial(u_func, local_node=j))

                        # Perform integration and add the result to the
                        # stiffness matrix.  Note that the integrand is the
                        # product of the functions in the functions tuple.
                        self.K[element.index+i][element.index+j] += \
                            Utils.integrate(functions, x_l, x_r,
                                            self.quad_points)

            # Assemble the right-hand side vector F if there is a source
            # function
            if self.weak_form.source_function:
                # The integral on the right-hand side is always of the form:
                #   \int W(x) f(x) dx
                for i in range(element.local_nodes):
                    # Create a tuple of constants and/or callables
                    functions = (partial(element.basis_function, local_node=i),
                                 self.weak_form.source_function)

                    # Perform integration and add the result to the right-hand
                    # side vector.
                    self.F[element.index+i] += \
                        Utils.integrate(functions, x_l, x_r, self.quad_points)

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
