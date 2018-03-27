import sys
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from fem1d.mesh import Mesh
from fem1d.model import Model


def p(x):
    nu = 2

    x = np.asarray(x)
    return nu * np.ones_like(x)


def q(x):
    x = np.asarray(x)
    return np.zeros_like(x)


def r(x):
    lam = 5

    x = np.asarray(x)
    return lam * np.ones_like(x)


def f(x):
    x = np.asarray(x)
    return np.zeros_like(x)


def u_exact(x):
    lam = 5
    nu = 2
    L = 1

    Pe = lam * L / nu

    return (np.exp(Pe * x / L) - 1) / (math.exp(Pe) - 1)


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
    #   The differential equation is defined for 0 < x < 1:
    #     
    #     -nu * u'' + lambda * u' = 0
    #
    #   with boundary conditions
    #
    #     u(0) = 0,
    #     u(1) = 1.
    #
    #   The exact solution is:
    #
    #     exact(x) = (exp(Pe * x / L) - 1) / (exp(Pe) - 1)
    #

    # Initialize variables that define the problem.

    num_elements = 10
    bc_type = 1
    quadrature_points = 2
    bc_left = 0.0
    bc_right = 1.0
    x_left = 0
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
