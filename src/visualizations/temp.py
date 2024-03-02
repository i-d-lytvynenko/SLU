import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

x = np.linspace(-10, 10, 100)
k_values = np.linspace(-0.5, 0.5, 10)
min_k, max_k = min(k_values), max(k_values)

plt.figure(figsize=(10, 6))

for k in k_values:
    y = k*x
    plt.plot(x, y, color=cm.plasma((k - min_k) / (max_k - min_k)))

scalarmap = cm.ScalarMappable(cmap='plasma')
scalarmap.set_array(k_values)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of the function k*x for different values of k')

cax = plt.axes([0.95, 0.1, 0.02, 0.8])
plt.colorbar(scalarmap, cax=cax, label='k')
cax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.show()
