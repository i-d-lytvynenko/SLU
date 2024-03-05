import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

from ..colors import COLORS
from ..layers import SLU
from ..types import Callable, Directory, Tuple
from ..utils import train
from .base import BaseExperiment, BasePreprocessor


np.random.seed(0)
torch.manual_seed(0)


class Preprocessor(BasePreprocessor):
    def fit(self, X: torch.Tensor) -> None:
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std


class Classify2D(BaseExperiment):
    datasets = {'spirals', 'moons'}

    def __init__(self, artifacts_dir: Directory, dataset: str,
                 n_samples: int = 2000, noise: float = .3,
                 n_epochs: int = 50, lr: float = 1e-3, is_verbose: bool = True):
        assert dataset in Classify2D.datasets
        self.dataset = dataset
        self.artifacts_dir = artifacts_dir
        self.preprocessor = Preprocessor()
        self.n_samples = n_samples
        self.noise = noise
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_verbose = is_verbose

    @staticmethod
    def make_spirals(n_samples: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.sqrt(np.random.rand(n_samples))*2*np.pi

        r_a = 2*theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + np.random.randn(n_samples, 2) * 3*noise

        r_b = -2*theta - np.pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + np.random.randn(n_samples, 2) * 3*noise

        res_a = np.append(x_a, np.zeros((n_samples, 1)), axis=1)
        res_b = np.append(x_b, np.ones((n_samples, 1)), axis=1)

        res = np.append(res_a, res_b, axis=0)
        np.random.shuffle(res)

        return res[:, :2], res[:, 2]

    @staticmethod
    def init_network(activation_fn: Callable[[int], nn.Module]) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2, 5),
            activation_fn(5),
            nn.Linear(5, 5),
            activation_fn(5),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def train(self) -> None:
        if self.dataset == 'spirals':
            X, y = Classify2D.make_spirals(
                n_samples=self.n_samples,
                noise=self.noise
            )
        else:
            X, y = make_moons(
                n_samples=self.n_samples,
                noise=self.noise
            )
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        train_loader, val_loader = self.get_data_loaders(X, y)

        relu_net = Classify2D.init_network(lambda _: nn.ReLU())
        slu_shared_net = Classify2D.init_network(lambda _: SLU())
        slu_ind_net = Classify2D.init_network(lambda size: SLU(size))

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

    def plot_classification(self) -> None:
        if self.dataset == 'spirals':
            X, y = Classify2D.make_spirals(
                n_samples=self.n_samples,
                noise=self.noise
            )
        else:
            X, y = make_moons(
                n_samples=self.n_samples,
                noise=self.noise
            )
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        X = self.preprocessor.transform(X)
        X = X.numpy()
        y = y.numpy()
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        grid_tensor = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

        if not hasattr(self, 'nets'):
            self.nets = torch.load(self.artifacts_dir/'nets.pth')

        for net_name, net in self.nets.items():
            with torch.no_grad():
                Z = net(grid_tensor)
                Z = torch.round(Z).numpy().reshape(xx.shape)

            plt.contourf(xx, yy, Z, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title(f'Decision Boundary — {net_name}')
            subfolder_name = self.artifacts_dir/(net_name.replace(" ", "_"))
            subfolder_name.mkdir(exist_ok=True)
            plt.savefig(subfolder_name/'classification.png')
            plt.clf()

    def plot_loss(self) -> None:
        ...
        plt.clf()

    def plot(self) -> None:
        self.plot_classification()
        self.plot_loss()
