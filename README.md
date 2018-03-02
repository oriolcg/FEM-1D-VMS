# FEM-1D

This project was born in my attempt to fully comprehend the mathematical foundations of the finite element method. The goal of this repository is to make it SIMPLE to LEARN how the finite element method works. This code is NOT intended to be fast and efficient.

The code is OBJECT-ORIENTED which should make it easy to read. The interface to the library is designed to be INTUITIVE: you just need to tell it about the WEAK FORM of the problem and its boundary conditions, specify the mesh, and tell it to compute the solution.

For now, the code is limited to 1D linear problems and piecewise linear basis functions. The code can solve ANY problem of the general form:

```a u'' + b u' + c u = f(x)```

where `a`, `b`, and `c` are constants. Dirichlet and Neumann boundary conditions are supported.

The code MIGHT be able to solve problems of the general form:

```a(x) u'' + b(x) u' + c(x) u = f(x)```

More testing is necessary before these problems are fully supported.

### Getting the code

1. Clone the repository:
```bash
git clone https://github.com/michelrobijns/FEM-1D
cd FEM-1D
```

2. Create a virtualenvironment and install the dependencies:
```bash
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

Check out the examples in the examples folder. More examples and explanation coming soon.

### Author

Michel Robijns

### Credits

I used the code on [this page](http://people.sc.fsu.edu/~jburkardt/py_src/fem1d/fem1d.html) to check my first implementation.
