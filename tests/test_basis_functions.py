import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from fem.element import LinearElement

x = np.linspace(0, 1, 10000)

el = LinearElement(0, 0.2, 0.4)

print(el.W_l(0.3))
print(el.dW_l(0.3))

plt.plot(x, el.basis_function(x, local_node=0))
plt.plot(x, el.basis_function(x, local_node=1))
plt.show()

plt.plot(x, el.basis_gradient(x, local_node=0))
plt.plot(x, el.basis_gradient(x, local_node=1))
plt.show()
