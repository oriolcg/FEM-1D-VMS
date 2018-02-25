# FEM-1D

This small project was born in my attempt to fully comprehend the mathematical foundations of the finite element method. The goal of this repository is to make it SIMPLE to LEARN how the finite element method works. The goal is NOT to make this code fast and efficient.

For now, the code is limited to 1D linear problems and simple linear basis functions. The code can solve ANY problem of the general form:

```a u'' + b u' + c u = f(x)```

where `a`, `b`, and `c` are constants. Dirichlet and Neumann boundary conditions are supported.

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
