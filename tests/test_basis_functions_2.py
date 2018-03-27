import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from fem.element_2 import Element

x = np.linspace(-1, 1, 10000)

el = Element(0, -1, 1)

print(el.basis_function(0.3, local_node=0))
print(el.basis_gradient(0.3, local_node=0))

plt.plot(x, el.basis_function(x, local_node=0))
plt.plot(x, el.basis_function(x, local_node=1))
plt.show()

plt.plot(x, el.basis_gradient(x, local_node=0))
plt.plot(x, el.basis_gradient(x, local_node=1))
plt.show()
