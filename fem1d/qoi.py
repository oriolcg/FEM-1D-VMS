import sys
import numpy as np
import math
from functools import partial
from fem1d.utils import Utils
from fem1d.quadrature_rule import QuadratureRule


class QoI(object):
    #
    # Discussion:
    #
    #  Computes a quantity of interest (QoI) of the form:
    #
    #    $$ Q(u) = int_\Omega  qFunc(x) * u(x) d\Omega $$
    #
    #  where \Omega is the full domain (further developments could consider the integral over a specific interval)
    #  and qFunc(x) a given functional

    def __init__(self, model, qFunc, u_exact, num_quad_points):
        self.model = model
        self.qFunc = qFunc
        self.u_exact = u_exact
        self.num_quad_points = num_quad_points

    def computeTau(self,element, x):
        # Pe = h * r / (  2 * p )
        # tau = h / ( sqrt(3.0) * r ) * min(1, Pe / sqrt(10) )
        Pe = element.h * self.model.r(x) / ( 2.0 * self.model.p(x) )
        h = element.h
        r = self.model.r(x)
        p = self.model.p(x)
        return element.h / ( 2.0 * r ) * ( 1.0 / math.tanh( Pe ) - 1.0 / Pe )
        #return h / (math.sqrt(3.0) * r) * min( 1.0, Pe / math.sqrt(10.0))
        #return min( h / r, h**2 / (8.0 * p) )

    def compute(self):

        self.value = 0.0
        self.value_exact = 0.0

        # Set Quadrature rule
        quad_rule = QuadratureRule( self.num_quad_points )

        # Loop over elements.

        for element in self.model.mesh.elements:

            e = element.index

            # Loop over quadrature points.
            
            for k in range(self.num_quad_points):
                
                # Get xi location of quadrature point.
                xi = quad_rule.xi_q[k]
                
                # Calculate x location of quadrature point.
                x = element.x_left + 0.5 * (1 + xi) * element.h
                
                # Calculate quadrature weight.
                w = quad_rule.w_q[k] * 0.5 * element.h

                # Interpolate solution at quadrature point.
                u = self.model.interpolate( element, k, self.num_quad_points )

                # Compute the integral:
                #    qFunc(x) * u(x)
                f = self.qFunc(x) * u
                f_e = self.qFunc(x) * self.u_exact(x)
                self.value += w * f
                self.value_exact += w * f_e

    def error_estimator(self):
        
        self.error_est = 0.0
        self.error_est_bound = 0.0

        # Set Quadrature rule
        quad_rule = QuadratureRule( self.num_quad_points )
        
        # Loop over elements.

        for element in self.model.mesh.elements:

            e = element.index

            # Loop over basis functions.
            
            for i in range(element.num_nodes):
                
                # Loop over quadrature points.
                
                for k in range(self.num_quad_points):
                    
                    # Get xi location of quadrature point
                    xi = quad_rule.xi_q[k]

                    # Calculate x location of quadrature point
                    x = element.x_left + 0.5 * (1 + xi) * element.h

                    # Calculate quadrature weight
                    w = quad_rule.w_q[k] * 0.5 * element.h

                    # Compute the values of the basis functions and
                    # its gradients at node I.
                    basis_i = element.basis_function(x=x, local_node=i)
                    basis_i_x = element.basis_gradient(x=x, local_node=i)

                    # Compute time-scale parameter
                    tau = self.computeTau(element,x)

                    # Compute the contribution of the integral of the stabilization term:
                    #    tau * Residual(u).
                    # with:
                    #    Residual(u) = f(x) - [ -d/dx ( p(x) du/dx ) + q(x) * u + r(x) * du/dx  ]
                    # Note that for linear elements the second order derivatives are zero
                    Residual = self.model.f(x) - ( self.model.q(x) * basis_i * self.model.u[e+i] + self.model.r(x) * basis_i_x * self.model.u[e+i] )
                    f = self.qFunc(x) * tau * Residual
                    self.error_est += w * f


                # Add jump contribution
                x = element.x_left + i * element.h
                tau = self.computeTau(element,x)

                # Loop over basis functions.
                grad_u = 0.0
                for j in range(element.num_nodes):
                    basis_j_x = element.basis_gradient(x=x, local_node=j)
                    grad_u +=  basis_j_x * self.model.u[e+j]

                # Compute jumps contribution
                #  jump = p(x) * du/dx * n
                jump = self.model.p(x) * grad_u * (-1)**(i+1)
                f = self.qFunc(x) * tau / element.h * 0.5 * jump
                self.error_est_bound += f
                self.error_est += f
                
                
                

            
