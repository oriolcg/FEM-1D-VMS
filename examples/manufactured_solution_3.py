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
    return np.cos(math.pi*x)


def analytical_solution(x):
    return 1/(math.pi**2) * np.cos(math.pi*x) + 2/(math.pi**2) * x - 1/(math.pi**2)


def main():
    # The PDE is defined for 0 < x < 1:
    #   PDE: -u'' = f
    # with boundary conditions
    #   u(0) = 0,
    #   u(1) = 0.
    #
    # The source function is:
    #   f(x) = cos(pi*x)
    #
    # The exact solution is:
    #   exact(x) = 1/(pi^2) * cos(pi*x) + 2/(pi^2) * x - 1/(pi^2)
    #
    # The weak form is:
    #   \int_0_L dW/dx du/dx dx =
    #   \int_0_L W f dx + natural bounary condition

    # Specify the mesh
    x_start      = 0
    x_end        = 1
    num_elements = 15
    mesh = Mesh.uniform_grid(x_start, x_end, num_elements)

    # Specify the weak form
    weak_form = WeakForm()
    weak_form.add_dw_du()
    weak_form.source_function = source_function

    # Add the boundary conditions
    weak_form.bc_l = BoundaryCondition.dirichlet(value=0)
    weak_form.bc_r = BoundaryCondition.dirichlet(value=0)

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
