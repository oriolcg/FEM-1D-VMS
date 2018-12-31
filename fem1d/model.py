import sys
import numpy as np
import math
from functools import partial
from fem1d.utils import Utils
from fem1d.quadrature_rule import QuadratureRule

class Model(object):
    #
    # Discussion:
    #
    #   Solves a nonlinear 1D boundary value problem.
    #
    #   The differential equation has the form:
    #
    #     -d/dx ( p(x) du/dx ) + q(x) * u + r(x) * du/dx = f(x)
    #
    #   The finite element method uses piecewise linear basis
    #   functions.
    #
    #   Here U is an unknown scalar function of X defined on the
    #   interval [XL, XR], and P, Q, and F are given functions of X.
    #

    def __init__(self, mesh, p, q, r, f, bc_type, bc_left, bc_right,
        num_quad_points=2, basis_function_order=2):
        #
        #
        #
        # Inputs
        #
        #     (fem1d.mesh.Mesh) mesh
        #         The mesh object.
        #
        #     (function) p
        #         The coefficient function P(X).
        #
        #     (function) q
        #         The coefficient function Q(X).
        #
        #     (function) f
        #         The source function of the differential equation.
        #
        #     (int) bc_type
        #         Declares what the boundary conditions are.
        #         1, at the left endpoint, U has the value U_LEFT,
        #            at the right endpoint, U has the value U_RIGHT.
        #         2, at the left endpoint, U has the value U_LEFT,
        #            at the right endpoint, U' has the value U_RIGHT.
        #         3, at the left endpoint, U' has the value U_LEFT,
        #            at the right endpoint, U has the value U_RIGHT.
        #         4, at the left endpoint, U' has the value U_LEFT,
        #            at the right endpoint, U' has the value U_RIGHT.
        #
        #     (float) bc_left
        #         The value of the boundary condition at X = X_LEFT.
        #
        #     (float) bc_right
        #         The value of the boundary condition at X = X_RIGHT.
        #
        #     (int) num_quad_points
        #         The number of quadrature points per element.
        #

        self.mesh = mesh
        self.p = p
        self.q = q
        self.r = r
        self.f = f
        self.bc_type = bc_type
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.num_quad_points = num_quad_points
        self.basis_function_order = basis_function_order
        self.K = None
        self.u = None
        self.F = None


    def solve(self):

        self.u = np.zeros(self.mesh.num_elements+1)

        self.assemble()

        self.__applyBC()

        self.u = np.linalg.solve(self.K, self.F)


    def assemble(self):

        self.K = np.zeros((self.mesh.num_elements+1, self.mesh.num_elements+1))
        self.F = np.zeros_like(self.u)

        # Set Quadrature rule
        quad_rule = QuadratureRule( self.num_quad_points )

        # Loop over elements.

        for element in self.mesh.elements:

            e = element.index

            # Loop over basis functions.

            for i in range(element.num_nodes):

                # Loop over basis functions.

                for j in range(element.num_nodes):

                    # Loop over quadrature points.

                    for k in range(self.num_quad_points):

                        # Get xi location of quadrature point.
                        xi = quad_rule.xi_q[k]

                        # Calculate x location of quadrature point.
                        x = element.x_left + 0.5 * (1 + xi) * element.h

                        # Calculate quadrature weight.
                        w = quad_rule.w_q[k] * 0.5 * element.h

                        # Compute the values of the basis functions and
                        # its gradients at nodes I and J.
                        basis_i = element.basis_function(x=x, local_node=i)
                        basis_j = element.basis_function(x=x, local_node=j)
                        basis_i_x = element.basis_gradient(x=x, local_node=i)
                        basis_j_x = element.basis_gradient(x=x, local_node=j)

                        # Compute the integral of the first term on the
                        # left of the PDE:
                        #     dW/dx * p * du/dx.

                        f = basis_i_x * self.p(x) * basis_j_x
                        self.K[e+i, e+j] += w * f

                        # Compute the integral of the second term on
                        # the left of the PDE:
                        #     W * p * u.

                        f = basis_i * self.q(x) * basis_j
                        self.K[e+i, e+j] += w * f

                        # Compute the integral of the third term on the
                        # left of the PDE:
                        #    W * r * du/dx.

                        f = basis_i * self.r(x) * basis_j_x
                        self.K[e+i, e+j] += w * f


            for i in range(element.num_nodes):

                for k in range(self.num_quad_points):

                    # Get xi location of quadrature point
                    xi = quad_rule.xi_q[k]

                    # Calculate x location of quadrature point
                    x = element.x_left + 0.5 * (1 + xi) * element.h

                    # Calculate quadrature weight
                    w = quad_rule.w_q[k] * 0.5 * element.h

                    basis_i = element.basis_function(x=x, local_node=i)

                    # Compute the integral of the term on the right of
                    # the PDE:
                    #     W * f(x).

                    f = basis_i * self.f(x)
                    self.F[e+i] += w * f

    def __applyBC(self):

        # Now that the stiffness matrix K and the vector F are
        # assembled, let us approach the boundary conditions.

        # Set left boundary condition
        if self.bc_type == 1 or self.bc_type == 2:

            # At the left endpoint, U has the value BC_LEFT

            self.F[0] = self.bc_left
            self.K[0, :] = 0
            self.K[0, 0] = 1
        else:

            # At the left endpoint, U' has the value BC_LEFT

            self.F[0] += -1 * self.bc_left

        # Set right boundary condition
        if self.bc_type == 1 or self.bc_type == 3:

            # At the right endpoint, U has the value BC_RIGHT

            self.F[-1] = self.bc_right
            self.K[-1, :] = 0
            self.K[-1, -1] = 1
        else:

            # At the right endpoint, U' has the value BC_RIGHT

            self.F[-1] += self.bc_right


    def interpolate(self, element, k, num_quad_points):

        e = element.index
        u = 0.0
        
        # Set Quadrature rule
        quad_rule = QuadratureRule( num_quad_points )

        # Loop over basis functions.
        for i in range(element.num_nodes):
            
            # Get xi location of quadrature point.
            xi = quad_rule.xi_q[k]

            # Calculate x location of quadrature point.
            x = element.x_left + 0.5 * (1 + xi) * element.h

            # Compute the values of the basis functions.
            basis_i = element.basis_function(x=x, local_node=i)
            
            # Interpolate solution
            u += basis_i * self.u[e+i]

        return u
