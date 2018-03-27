import sys
import numpy as np
import matplotlib.pyplot as plt


def basis_function(node, xi, order):
    # PROTOTYPE
    basis_function_order = order  # 1: linear, 2: quadratic, 3: cubic
    xi_at_node = np.linspace(-1, 1, basis_function_order+1)
    # END PROTOTYPE

    # Vectorization
    xi = np.asarray(xi)
    value = np.ones_like(xi)

    A = node

    # for i in range(value.size):
    #     for B in range(basis_function_order+1):
    #         if B != A:
    #             value[i] *= (xi[i] - xi_at_node[B]) / \
    #                         (xi_at_node[A] - xi_at_node[B])

    for B in range(basis_function_order+1):
        if B != A:
            value *= (xi - xi_at_node[B]) / \
                     (xi_at_node[A] - xi_at_node[B])

    return value


def basis_gradient(node, xi, order):
    # PROTOTYPE
    basis_function_order = order  # 1: linear, 2: quadratic, 3: cubic
    xi_at_node = np.linspace(-1, 1, basis_function_order+1)
    # END PROTOTYPE

    # Vectorization
    xi = np.asarray(xi)
    value = np.zeros_like(xi)

    A = node

    for B in range(basis_function_order+1):
        if B != A:
            value += 1.0 / (xi - xi_at_node[B])

    value *= basis_function(node, xi, order)

    return value


def basis_gradient_2(node, xi, order):
    # PROTOTYPE
    basis_function_order = order  # 1: linear, 2: quadratic, 3: cubic
    xi_at_node = np.linspace(-1, 1, basis_function_order+1)
    # END PROTOTYPE

    # Vectorization
    xi = np.asarray(xi)
    value = np.zeros_like(xi)
    temp_value = np.empty_like(xi)

    A = node

    for l in range(basis_function_order+1):
        if l != A:
            temp_value = 1.0 / (xi_at_node[A] - xi_at_node[l])

            for m in range(basis_function_order+1):
                if m != A and m != l:
                    temp_value *= (xi - xi_at_node[m]) / \
                                  (xi_at_node[A] - xi_at_node[m])

            value += temp_value

    return value


def main():
    order = 10

    xi = np.linspace(-1, 1, 2000)

    #print(basis_function(node=0, xi=-1.0, order=order))
    #print(basis_function(node=0, xi=-0.724, order=order))
    #print(basis_gradient(node=0, xi=-0.724))

    for i in range(order+1):
        plt.plot(xi, basis_function(node=i, xi=xi, order=order))

    plt.show()

    # plt.plot(xi, basis_function(node=0, xi=xi, order=order))
    # #plt.plot(xi, basis_function(node=1, xi=xi, order=order))
    # #plt.plot(xi, basis_function(node=2, xi=xi, order=order))
    # #plt.plot(xi, basis_gradient_2(node=0, xi=xi, order=order))
    # plt.plot([-1, 1], [0, 0], "k:")
    # plt.show()


if __name__ == '__main__':
    main()
