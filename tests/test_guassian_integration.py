import sys
sys.path.insert(0, "..")
from fem1d.utils import Utils


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
    print(Utils.integrate((h,h), 0, 10, 4))
    print(Utils.integrate((h,h,h), 0, 10, 5))


if __name__ == '__main__':
    main()
