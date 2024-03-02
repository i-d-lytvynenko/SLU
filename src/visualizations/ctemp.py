# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.gridspec as gridspec

# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# fig = plt.figure(figsize=(12, 4))
# gs = gridspec.GridSpec(1, 3, width_ratios=[4, 1, 4])

# ax1 = plt.subplot(gs[0])
# ax1.plot(x, y, color='blue')
# ax1.set_title('Figure 1')

# ax2 = plt.subplot(gs[2])
# sc = ax2.scatter(x, y, c=y, cmap='viridis')
# ax2.set_title('Figure 2')

# cb_ax = fig.add_axes([0.48, 0.15, 0.04, 0.7])
# cb = plt.colorbar(sc, cax=cb_ax, orientation='vertical')

# plt.show()




import matplotlib as mpl
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
col_map = plt.get_cmap('nipy_spectral')
mpl.colorbar.ColorbarBase(ax, cmap=col_map, orientation = 'vertical')

# # As for a more fancy example, you can also give an axes by hand:
# c_map_ax = fig.add_axes([0.2, 0.8, 0.6, 0.02])
# c_map_ax.axes.get_xaxis().set_visible(False)
# c_map_ax.axes.get_yaxis().set_visible(False)

# # and create another colorbar with:
# mpl.colorbar.ColorbarBase(c_map_ax, cmap=col_map, orientation = 'horizontal')
plt.show()
