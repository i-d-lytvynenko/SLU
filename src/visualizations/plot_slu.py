from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FormatStrFormatter
import torch


def relu(x):
    return torch.max(torch.tensor(0), x)


def slu(x, k=0.0):
    A = torch.log(1 + torch.abs(x))
    B = k * A.pow(2)
    return torch.where(x > 0, x + B, B - A)


@dataclass
class PlotSLU:
    k_min: float = -0.5
    k_max: float = 0.5
    x_min: float = -20.
    x_max: float = 20.

    def evaluate(self):
        X = torch.linspace(self.x_min, self.x_max, 500, requires_grad=True)
        X_arr = X.detach().numpy()
        k_values = np.linspace(self.k_min, self.k_max, 120)

        _, axs = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'width_ratios': [48, 4, 48]})
        axs[0].set_title('SLU for different values of k')
        axs[1].set_title('k')
        axs[2].set_title('Derivative of SLU for different values of k')

        axs[0].set_xlim([self.x_min, self.x_max])
        axs[0].axhline(y=0, color='k', alpha=0.8)
        axs[0].axvline(x=0, color='k', alpha=0.8)

        axs[2].set_xlim([self.x_min, self.x_max])
        axs[2].plot(X_arr, X_arr*0, color='k', alpha=0.8)
        axs[2].plot(X_arr, X_arr*0 + 1, color='g', linewidth=10, alpha=0.2)
        axs[2].plot(X_arr, X_arr*0, color='r', linewidth=10, alpha=0.2)
        axs[2].axvline(x=0, color='k', alpha=0.8)

        for k in k_values:
            Y = slu(X, k)
            Y.backward(torch.ones_like(X), retain_graph=True)
            Y_prime = X.grad
            k_norm = (k - self.k_min) / (self.k_max - self.k_min)
            axs[0].plot(X_arr, Y.detach().numpy(), alpha=0.4, color=cm.plasma(k_norm))
            axs[2].plot(X_arr, Y_prime.detach().numpy(), alpha=0.4, color=cm.plasma(k_norm))
            X.grad.zero_()

        axs[0].grid(alpha=0.4)
        axs[2].grid(alpha=0.4)
        ColorbarBase(axs[1], cmap='plasma',
                     ticks=np.linspace(self.k_min, self.k_max, 5),
                     values=np.linspace(self.k_min, self.k_max, 100),
                     format=FormatStrFormatter('%.2f'), ticklocation='left')
        plt.tight_layout()
