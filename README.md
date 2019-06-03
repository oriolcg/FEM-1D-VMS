# FEM-1D-VMS

1D Finite Element code for the Convection-Diffusion-Reaction equation with Variational Multiscale stabilization. Code forked from https://github.com/michelrobijns/FEM-1D

# FEM-1D

The goal of this repository is to make it simple to learn how the finite element method works. This code is not intended to be fast and efficient. The code is object-oriented which should make it easy to read and the interface is designed to be intuitive.

For now, the code is limited to 1D linear problems and piecewise linear basis functions. The code can solve any problem of the form:

```-d/dx ( p(x) * du/dx ) + q(x) * u + r(x) * du/dx = f(x)```

Here U is an unknown scalar function of X defined on the interval [X_LEFT, X_RIGHT]. Both Dirichlet and Neumann boundary conditions are supported.

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

Check out the examples in the examples folder. More examples and are coming soon.

### Author

Michel Robijns

### Credits

I used the code on [this page](http://people.sc.fsu.edu/~jburkardt/py_src/fem1d/fem1d.html) to check my first implementation.
