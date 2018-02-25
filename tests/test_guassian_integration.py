import sys
sys.path.insert(0, "..")
from fem.utils import Utils


def f(x):
    return x


def g(x):
    return x * x


def h(x):
    return x * x * x


def main():
    print(Utils.integrate(f, 0, 10, 3))
    print(Utils.integrate(g, 0, 10, 3))
    print(Utils.integrate(h, 0, 10, 3))
    print(Utils.integrate((f, g), 0, 10, 3))
    print(Utils.integrate((f, f, f), 0, 10, 3))


if __name__ == '__main__':
    main()
