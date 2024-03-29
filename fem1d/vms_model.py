import sys
import numpy as np
import math
from functools import partial
from fem1d.utils import Utils
from fem1d.model import Model
from fem1d.quadrature_rule import QuadratureRule


class VMSModel(Model):

    def __init__(self, mesh, p, q, r, f, bc_type, bc_left, bc_right, num_quad_points=2, basis_function_order=2):
        Model.__init__(self,mesh, p, q, r, f, bc_type, bc_left, bc_right, num_quad_points, basis_function_order)

    def solve(self):
        Model.solve(self)

    def computeTau(self,element, x):
        # Pe = h * r / (  2 * p )
        # tau = h / (2 * r) * ( coth(Pe) - 1/Pe )
        Pe = element.h * self.r(x) / ( 2.0 * self.p(x) )
        return element.h / ( 2.0 * self.r(x) ) * ( 1.0 / math.tanh( Pe ) - 1.0 / Pe )


    def assemble(self):
        Model.assemble(self)

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

                        # Compute time-scale parameter
                        tau = self.computeTau(element,x)

                        # Compute the LHS contribution of the integral of the stabilization term:
                        #    tau * Residual(u) * Ladj(W).
                        # with:
                        #    Ladj(w) = -d/dx ( p(x) dw/dx ) + q(x) * u - r(x) * du/dx   
                        #    Residual(u) = f(x) - [ -d/dx ( p(x) du/dx ) + q(x) * u + r(x) * du/dx  ]
                        # Note that for linear elements the second order derivatives are zero
                        Ladj = self.q(x) * basis_i - self.r(x) * basis_i_x
                        Residual = - ( self.q(x) * basis_j + self.r(x) * basis_j_x )
                        f = Ladj * tau * Residual
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
                    basis_i_x = element.basis_gradient(x=x, local_node=i)

                    # Compute time-scale parameter
                    tau = self.computeTau(element,x)

                    # Compute the LHS contribution of the integral of the stabilization term:
                    #    tau * Residual(u) * Ladj(W).
                    # with:
                    #    Ladj(w) = -d/dx ( p(x) dw/dx ) + q(x) * u - r(x) * du/dx   
                    #    Residual(u) = f(x) - [ -d/dx ( p(x) du/dx ) + q(x) * u + r(x) * du/dx  ]
                    # Note that for linear elements the second order derivatives are zero
                    Ladj = self.q(x) * basis_i - self.r(x) * basis_i_x
                    Residual = self.f(x)
                    f = Ladj * tau * Residual
                    self.F[e+i] += w * f
