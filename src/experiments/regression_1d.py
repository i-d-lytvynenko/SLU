from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator

from ..colors import COLORS
from ..layers import SLU
from ..types import Callable, Directory, Tuple
from ..utils import train, smooth_data, to_scientific_notation
from .base import BaseExperiment, BasePreprocessor


@dataclass
class Dataset1D:
    name: str
    latex: str
    range: Tuple[float, float]
    function: Callable


class Preprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.x_max = torch.max(X)
        self.x_min = torch.min(X)
        self.y_max = torch.max(y)
        self.y_min = torch.min(y)

    def transform(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_norm = (X - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        return X_norm, y_norm


class Regress1D(BaseExperiment):
    datasets = [
        Dataset1D('square', 'x^2', (-5, 5), lambda x: x**2),
        Dataset1D('root', r'\sqrt{x}', (0, 5), lambda x: x**0.5),
        Dataset1D('reciprocal', r'\frac{1}{x}', (1, 5), lambda x: x**(-1)),
    ]

    def __init__(self, artifacts_dir: Directory, dataset: str,
                 n_samples: int = 2000, noise: float = .3,
                 n_epochs: int = 100, lr: float = 1e-3,
                 is_verbose: bool = True, window_size: int = 40):
        super().__init__(artifacts_dir, Preprocessor)
        self.dataset: Dataset1D = next((ds for ds in Regress1D.datasets if ds.name == dataset), None)
        if self.dataset is None:
            raise ModuleNotFoundError('No dataset with such name exists')
        self.n_samples = n_samples
        self.noise = noise
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_verbose = is_verbose
        self.window_size = window_size

    @staticmethod
    def make_data(dataset: Dataset1D, n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        X = np.linspace(*dataset.range, n_samples)[:, None]
        y = dataset.function(X)
        y += np.random.randn(n_samples, 1) * 0.1*noise
        return X, y

    @staticmethod
    def init_network(activation_fn: Callable[[int], nn.Module]) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, 5),
            activation_fn(5),
            nn.Linear(5, 5),
            activation_fn(5),
            nn.Linear(5, 1),
        )

    def train(self) -> None:
        X, y = Regress1D.make_data(
            dataset=self.dataset,
            n_samples=self.n_samples,
            noise=self.noise
        )
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        train_loader, val_loader = self.get_data_loaders(X, y)

        relu_net = Regress1D.init_network(lambda _: nn.ReLU())
        slu_shared_net = Regress1D.init_network(lambda _: SLU())
        slu_ind_net = Regress1D.init_network(lambda size: SLU(size))

        self.nets = {
            'ReLU': relu_net,
            'SLU (shared)': slu_shared_net,
            'SLU (individual)': slu_ind_net,
        }

        for net_name, net in self.nets.items():
            train_loss_log, val_loss_log = train(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.MSELoss(),
                lr=self.lr,
                n_epochs=self.n_epochs,
                is_verbose=self.is_verbose
            )
            subfolder_name = self.artifacts_dir/(net_name.replace(" ", "_"))
            subfolder_name.mkdir(exist_ok=True)
            torch.save(train_loss_log, subfolder_name/'train_loss.pth')
            torch.save(val_loss_log, subfolder_name/'val_loss.pth')

        torch.save(self.nets, self.artifacts_dir/f'nets.pth')

    def plot_regression(self) -> None:
        X, y = Regress1D.make_data(
            dataset=self.dataset,
            n_samples=self.n_samples,
            noise=self.noise
        )
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if not self.preprocessor.is_trained:
            self.preprocessor = torch.load(self.artifacts_dir/'preprocessor.pth')
        X_tensor, y_tensor = self.preprocessor.transform(X_tensor, y_tensor)
        X = X_tensor.numpy()
        y = y_tensor.numpy()

        if not hasattr(self, 'nets'):
            self.nets = torch.load(self.artifacts_dir/'nets.pth')

        for net_name, net in self.nets.items():
            with torch.no_grad(): y_ = net(X_tensor)
            plt.figure(figsize=(8, 6))
            plt.scatter(X, y, alpha=0.8, label=f'exact (noise = {self.noise})', c='k')
            plt.scatter(X, y_, alpha=0.8, label='approximate', c=COLORS[net_name])
            plt.xlabel('x (normalized)')
            plt.ylabel('y (normalized)')
            plt.legend()
            plt.title(f'$y={self.dataset.latex}$ approximation — {net_name}')
            plt.tight_layout()
            subfolder_name = self.artifacts_dir/(net_name.replace(' ', '_'))
            subfolder_name.mkdir(exist_ok=True)
            plt.savefig(subfolder_name/'approximation.png')
            plt.clf()

    def plot_loss(self) -> None:
        _, ax = plt.subplots(figsize=(8, 6))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for net_name in ['ReLU', 'SLU (shared)', 'SLU (individual)']:
            subfolder_name = self.artifacts_dir/(net_name.replace(' ', '_'))
            train_loss = torch.load(subfolder_name/'train_loss.pth')
            train_loss = smooth_data(train_loss, self.window_size)
            x_train = np.linspace(0, self.n_epochs, len(train_loss))
            plt.plot(x_train, train_loss,
                     c=COLORS[net_name], label=f'{net_name} - train', alpha=0.7)
            val_loss = torch.load(subfolder_name/'val_loss.pth')
            val_loss = smooth_data(val_loss, self.window_size)
            plt.tight_layout()
            x_val = np.linspace(0, self.n_epochs, len(val_loss))
            plt.plot(x_val, val_loss,
                     c=COLORS[net_name], label=f'{net_name} - val')
        plt.xlim([0, self.n_epochs])
        plt.xlabel('Epoch')
        plt.ylabel(f'MSE Loss (lr = {to_scientific_notation(self.lr)})')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.artifacts_dir/'train_comparison.png')
        plt.clf()

    def plot(self) -> None:
        self.plot_regression()
        self.plot_loss()
