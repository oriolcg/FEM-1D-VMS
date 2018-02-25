import sys
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from fem.mesh import Mesh
from fem.weakform import WeakForm
from fem.boundarycondition import BoundaryCondition
from fem.model import Model


def analytical_solution(x):
    lam = 5
    nu = 1
    L = 1

    Pe = lam * L / nu

    return (np.exp(Pe * x / L) - 1) / (math.exp(Pe) - 1)


def main():
    # The PDE is defined for 0 < x < 1:
    #   PDE: lambda * u' - nu * u'' = 0
    # with boundary conditions
    #   u(0) = 0,
    #   u(L) = 1.
    #
    # The exact solution is:
    #   exact(x) = (exp(Pe*x/L) - 1) / (exp(Pe) - 1)
    # where
    #   Pe = U * L / nu

    # Physical parameters
    lam = 5
    nu = 1
    L = 1

    # Specify the mesh
    x_start      = 0
    x_end        = L
    num_elements = 10
    mesh = Mesh.non_uniform_grid(x_start, x_end, num_elements, ratio=0.7)
    #mesh = Mesh.uniform_grid(x_start, x_end, num_elements)

    # Specify the weak form
    weak_form = WeakForm()
    weak_form.add_dw_du(nu)
    weak_form.add_w_du(lam)
    # The right-hand side is set to zero when there is no source function, so
    # there is nothing to specify for the right-hand side

    # Add the boundary conditions
    weak_form.bc_l = BoundaryCondition.dirichlet(value=0)
    weak_form.bc_r = BoundaryCondition.dirichlet(value=1)

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
