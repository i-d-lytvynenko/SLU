import matplotlib.pyplot as plt
import torch

from ..colors import COLORS
from ..types import File


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.tensor(0), x)


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return torch.where(x > 0, x, alpha*(torch.exp(x) - 1))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * x**3)))


def slu(x: torch.Tensor, k: float = 0.0) -> torch.Tensor:
    A = torch.log(1 + torch.abs(x))
    B = k * A.pow(2)
    return torch.where(x > 0, x + B, B - A)


class PlotActivations:
    def __init__(self, fig_path: File, x_min: float = -5., x_max: float = 5.):
        self.fig_path = fig_path
        self.x_min = x_min
        self.x_max = x_max

    def evaluate(self):
        X = torch.linspace(self.x_min, self.x_max, 500, requires_grad=True)
        X_arr = X.detach().numpy()
        function_info = {
            'ReLU': [relu, []],
            'ELU': [elu, []],
            'GELU': [gelu, []],
            'SLU (k = -0.2)': [slu, [-0.2]],
            'SLU (k = 0)': [slu, [0]],
            'SLU (k = 0.2)': [slu, [0.2]],
        }

        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].set_title('Functions')
        axs[1].set_title('Derivatives')

        y_lim = relu(torch.tensor(self.x_max)).numpy()
        axs[0].set_xlim([self.x_min, self.x_max])
        axs[0].set_ylim([-y_lim, y_lim])
        axs[0].axhline(y=0, color='k', alpha=0.8)
        axs[0].axvline(x=0, color='k', alpha=0.8)

        axs[1].set_xlim([self.x_min, self.x_max])
        axs[1].plot(X_arr, X_arr*0, color='k', alpha=0.8)
        axs[1].plot(X_arr, X_arr*0 + 1, color='g', linewidth=10, alpha=0.2)
        axs[1].plot(X_arr, X_arr*0, color='r', linewidth=10, alpha=0.2)
        axs[1].axvline(x=0, color='k', alpha=0.8)

        for i, (label, (f, params)) in enumerate(function_info.items()):
            Y = f(X, *params)
            Y.backward(torch.ones_like(X), retain_graph=True)
            Y_prime = X.grad
            ls = ['-', '--', '-.', ':'][i % 4]
            lw = [3, 4, 5, 6][i % 4]
            axs[0].plot(X_arr, Y.detach().numpy(), alpha=0.8,
                        c=COLORS[label], label=label, ls=ls, lw=lw)
            axs[1].plot(X_arr, Y_prime.detach().numpy(), alpha=0.8,
                        c=COLORS[label], label=label, ls=ls, lw=lw)
            X.grad.zero_()

        axs[0].legend()
        axs[0].grid(alpha=0.4)
        axs[1].legend()
        axs[1].grid(alpha=0.4)
        plt.tight_layout()

        plt.savefig(self.fig_path)
        plt.clf()
