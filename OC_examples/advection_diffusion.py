import sys
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from fem1d.mesh import Mesh
from fem1d.model import Model
from fem1d.vms_model import VMSModel
from fem1d.qoi import QoI


def p(x):
    x = np.asarray(x)
    return nu * np.ones_like(x)


def q(x):
    x = np.asarray(x)
    return react * np.ones_like(x)


def r(x):
    x = np.asarray(x)
    return lam * np.ones_like(x)


def f(x):
    x = np.asarray(x)
    return source * np.ones_like(x)


def u_exact(x):

    # y(x) = c1 + c2* exp(a*x) + x/b
    # y(0) = 0 = c1 + c2 --> c1 =-c2
    # y(1) = 0 = c2 ( exp(a*x) - 1 ) + 1/b --> c2 = 1 / (( 1 - exp(a)) * b)

    a = lam / nu
    b = lam
    c2 = 1.0 / (( 1.0 - math.exp(a)) * b)
    c1 = -c2
    return c1 + c2 * np.exp(a*x) + x/b

def qoiFunc(x):
    x = np.asarray(x)
    return 1.0 / L * np.ones_like(x)
    

def main():
    #
    # Discussion:
    #
    #   Solves a linear 1D boundary value problem.
    #
    #   The differential equation has the form:
    #
    #     -d/dx ( p(x) du/dx ) + q(x) * u + r(x) * du/dx = f(x)
    #
    #   The finite element method uses piecewise linear basis
    #   functions and a Variational Multiscale stabilization.
    #
    #   Here U is an unknown scalar function of X defined on the
    #   interval [XL, XR], and P, Q, R, and F are given functions of X.
    #
    #
    #   The differential equation is defined for 0 < x < 1:
    #     
    #     -nu * u'' + lambda * u' = 1
    #
    #   with boundary conditions
    #
    #     u(0) = 0,
    #     u(1) = 0.
    #
    #   The exact solution is:
    #     u(x) = c1 + c2 * exp( lambda * x / nu ) + x / lambda
    #     c1 = -c2
    #     c2 = 1 / (( 1 - exp( lambda / nu )) * lambda )
    #

    # Initialize variables that define the problem.
    # =============================================
    # Mesh properties
    num_elements = 10
    bc_type = 1
    quadrature_points = 2
    bc_left = 0.0
    bc_right = 0.0
    x_left = 0
    x_right = 1
    global L
    L = x_right - x_left

    # Physical properties
    global lam
    global nu
    global react
    global source
    lam = 1.0
    nu = 1.0e-2
    react = 0.0
    source = 1.0

    # Compute Peclet number
    h = L / float(num_elements)
    peclet = h * lam / (2.0 * nu)
    print 'The local Peclet number is: ', peclet

    # Create the mesh object that describes the geometry of the problem.
    # ==================================================================
    mesh = Mesh.uniform_grid(x_left, x_right, num_elements)

    # Create the Galerkin model object.
    # =================================
    gal_model = Model(mesh, p, q, r, f, bc_type, bc_left, bc_right, quadrature_points)
    vms_model = VMSModel(mesh, p, q, r, f, bc_type, bc_left, bc_right, quadrature_points)

    # Solve the Galerkin problem.
    # ===========================
    gal_model.solve()
    vms_model.solve()

    # Compute the exact solution
    # ==========================
    u_e = u_exact(mesh.x)
    gal_error = np.abs(u_e - gal_model.u)
    vms_error = np.abs(u_e - vms_model.u)
    
    # Compute Quantity of Interest
    # ============================
    qoi_gal = QoI(gal_model, qoiFunc, u_exact, 10)
    qoi_vms = QoI(vms_model, qoiFunc, u_exact, 10)
    qoi_gal.compute()
    qoi_vms.compute()
    qoi_vms.error_estimator()
    
    # Print nodal error
    # =================
    '''
    print("")
    print("        X           U_Gal(X)      U_VMS(X)      U(exact)      Error_Gal     Error_VMS")
    print("")
    for i in range(0, num_elements + 1):
        print("  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}".format(mesh.x[i], gal_model.u[i], vms_model.u[i], u_e[i], gal_error[i], vms_error[i]))
    '''

    # Print QoI value
    # ===============
    print("")
    print ' Quantity of Interest values'
    print ' ==========================='
    print ' Galerkin:   ', qoi_gal.value
    print ' VMS:        ', qoi_vms.value
    print ' Exact:      ', qoi_gal.value_exact
    print ' Error_Gal:  ', abs(qoi_gal.value_exact - qoi_gal.value )
    print ' Error_VMS:  ', abs(qoi_vms.value_exact - qoi_vms.value )
    print ' Error_est:  ', abs(qoi_vms.error_est)
    print ' Efficiency: ', abs(qoi_vms.error_est) / abs(qoi_vms.value_exact - qoi_vms.value )
        
    # Plot solution
    # =============
    fine_mesh = Mesh.uniform_grid(x_left, x_right, num_elements*100)
    plt.plot(fine_mesh.x, u_exact(fine_mesh.x), "k", linewidth=1)
    plt.plot(gal_model.mesh.x, gal_model.u, "k*-", linewidth=1)
    plt.plot(vms_model.mesh.x, vms_model.u, "ks-", linewidth=1)
    #plt.savefig("fem_results.eps")
    #plt.show()


if __name__ == '__main__':
    main()
