import sys
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from fem1d.mesh import Mesh
from fem1d.model import Model


def p(x):
    x = np.asarray(x)
    return -np.ones_like(x)


def q(x):
    x = np.asarray(x)
    return np.zeros_like(x)


def r(x):
    x = np.asarray(x)
    return np.zeros_like(x)


def f(x):
    return -50 * np.exp(x)


def u_exact(x):
    return -50 * np.exp(x) + 50 * x * math.sinh(1) + 100 + 50 * math.cosh(1)


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
    #   functions.
    #
    #   Here U is an unknown scalar function of X defined on the
    #   interval [XL, XR], and P, Q, R, and F are given functions of X.
    #
    #
    #   The differential equation is defined for -1 < x < 1:
    #     
    #     u'' = f
    #
    #   with boundary conditions
    #
    #     u(-1) = 0,
    #     u(1) = 0.
    #
    #   The source function is:
    #
    #     f(x) = -50 * exp(x)
    #
    #   The exact solution is:
    #
    #     exact(x) = -50 * exp(x) + 50 * x * sinh(1) + 100 + 50
    #       * cosh(1)
    #

    # Initialize variables that define the problem.

    num_elements = 10
    bc_type = 1
    quadrature_points = 2
    bc_left = 100.0
    bc_right = 100.0
    x_left = -1
    x_right = 1

    # Create the mesh object that describes the geometry of the problem.

    mesh = Mesh.uniform_grid(x_left, x_right, num_elements)

    # Create the model object.

    model = Model(mesh, p, q, r, f, bc_type, bc_left, bc_right, 
        quadrature_points)

    # Solve the problem.

    model.solve()

    # Compute the exact solution

    u_e = u_exact(mesh.x)
    error = np.abs(u_e - model.u)

    # Print nodal error

    print("")
    print("        X             U(X)          U(exact)      Error")
    print("")

    for i in range(0, num_elements + 1):
        print("  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}".format(mesh.x[i], \
            model.u[i], u_e[i], error[i]))

    # Plot solution
    fine_mesh = Mesh.uniform_grid(x_left, x_right, num_elements*100)
    plt.plot(fine_mesh.x, u_exact(fine_mesh.x), "k", linewidth=1)
    plt.plot(model.mesh.x, model.u, "k*-", linewidth=1)
    #plt.savefig("fem_results.eps")
    plt.show()


if __name__ == '__main__':
    main()
