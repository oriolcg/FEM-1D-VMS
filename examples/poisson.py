import sys
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from fem.mesh import Mesh
from fem.weakform import WeakForm
from fem.boundarycondition import BoundaryCondition
from fem.model import Model


def source_function(x):
    return -50 * np.exp(x)


def analytical_solution(x):
    return -50 * np.exp(x) + 50 * x * math.sinh(1) + 100 + 50 * math.cosh(1)


def main():
    # The PDE is defined for -L < x < L:
    #   PDE: u'' = f
    # with boundary conditions
    #   u(-L) = 100,
    #   u(L) = 100.
    #
    # The source function is:
    #   f(x) = 50 * exp(x)
    #
    # The exact solution is:
    #   exact(x) = -50 * exp(x) + 50 * x * sinh(1) + 100 + 50 * cosh(1)
    #
    # The weak form is:
    #   - \int_0_L dW/dx du/dx dx =
    #   \int_0_L W f dx - natural bounary condition

    # Specify the mesh
    x_start      = -1
    x_end        = 1
    num_elements = 10
    mesh = Mesh.uniform_grid(x_start, x_end, num_elements)

    # Specify the weak form
    weak_form = WeakForm()
    weak_form.add_dw_du(constant=-1)
    weak_form.source_function = source_function

    # Add the boundary conditions
    weak_form.bc_l = BoundaryCondition.dirichlet(value=100)
    weak_form.bc_r = BoundaryCondition.dirichlet(value=100)

    # Create the model object
    model = Model(mesh, weak_form)
    model.solve()

    u_analytical = analytical_solution(mesh.x)
    error = np.abs(u_analytical - model.u)

    # Print nodal error
    print("")
    print("  Node          Ucomp           Uexact          Error")
    print("")

    for i in range(0, num_elements + 1):
        print("{:4d}  {:14.6g}  {:14.6g}  {:14.6g}".format(i, model.u[i],
              u_analytical[i], error[i]))

    # Plot solution
    fine_mesh = Mesh.uniform_grid(x_start, x_end, num_elements*100)
    plt.plot(fine_mesh.x, analytical_solution(fine_mesh.x), "k", linewidth=1)
    plt.plot(model.mesh.x, model.u, "k*-", linewidth=1)
    #plt.savefig("fem_results.eps")
    plt.show()


if __name__ == '__main__':
    main()
