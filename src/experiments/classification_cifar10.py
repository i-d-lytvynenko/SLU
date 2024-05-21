from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import torch.utils.data as data
from torchvision import datasets, transforms

from ..colors import COLORS
from ..layers import SLU
from ..types import Callable, Directory, Tuple, Literal
from ..utils import smooth_data, to_scientific_notation, train
from .base import BaseExperiment, BasePreprocessor


class Preprocessor(BasePreprocessor):
    '''NOTE:
    No preprocessor is needed
    as pytorch has a prebuilt CIFAR10 integration
    with torchvision transforms support
    '''
    def __init__(self):
        super().__init__()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def transform(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return X, y


class ClassifyCIFAR10(BaseExperiment):
    def __init__(self, artifacts_dir: Directory,
                 force_retrain: bool,
                 network_type: Literal['FC', 'CNN'],
                 n_layers: int = 4, n_neurons: int = 64,
                 noise: float = 0,
                 n_epochs: int = 20, lr: float = 1e-3,
                 is_verbose: bool = True, window_size: int = 128):
        super().__init__(artifacts_dir, Preprocessor)
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.force_retrain = force_retrain
        self.network_type = network_type
        self.noise = noise
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_verbose = is_verbose
        self.window_size = window_size

    def init_network(self, activation_fn: Callable[[int], nn.Module]) -> nn.Sequential:
        if self.network_type == 'FC':
            layers = [
                nn.Linear(3*32*32, self.n_neurons),
                activation_fn(self.n_neurons)
            ]
            for _ in range(self.n_layers - 1):
                layers.append(nn.Linear(self.n_neurons, self.n_neurons))
                layers.append(activation_fn(self.n_neurons))
            layers.append(nn.Linear(self.n_neurons, 10))
            return nn.Sequential(*layers)
        else:
            return nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3, padding=1),
                activation_fn(32),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                activation_fn(32),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                activation_fn(32),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=0.5),

                nn.Conv2d(96, 192, kernel_size=3, padding=1),
                activation_fn(16),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                activation_fn(16),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                activation_fn(16),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=0.5),

                nn.Conv2d(192, 192, kernel_size=3),
                activation_fn(6),
                nn.Conv2d(192, 192, kernel_size=1),
                activation_fn(6),
                nn.Conv2d(192, 192, kernel_size=1),
                activation_fn(6),
                nn.AvgPool2d(kernel_size=6),
                nn.Flatten(),
                nn.Linear(192, 10),
            )

    def train(self) -> None:
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x + torch.randn(x.shape) * self.noise)
        ]
        if self.network_type == 'FC':
            transformations.append(transforms.Lambda(torch.flatten))
        transform = transforms.Compose(transformations)

        train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
        train_loader = data.DataLoader(train_dataset, batch_size=128)
        val_loader = data.DataLoader(val_dataset, batch_size=128)

        relu_net = self.init_network(lambda _: nn.ReLU())
        elu_net = self.init_network(lambda _: nn.ELU())
        gelu_net = self.init_network(lambda _: nn.GELU())
        slu_shared_net = self.init_network(lambda _: SLU())
        slu_ind_net = self.init_network(lambda size: SLU(size))

        self.nets = {
            'ReLU': relu_net,
            'ELU': elu_net,
            'GELU': gelu_net,
            'SLU (shared)': slu_shared_net,
            'SLU (individual)': slu_ind_net,
        }

        for net_name, net in self.nets.items():
            subfolder_name = self.artifacts_dir/(net_name.replace(' ', '_'))
            if not self.force_retrain and subfolder_name.exists():
                if self.is_verbose:
                    print(f'{net_name} is pretrained, moving on')
                continue

            if self.is_verbose:
                print(f'{net_name} training started')

            train_loss_log, val_loss_log = train(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                lr=self.lr,
                n_epochs=self.n_epochs,
                is_verbose=self.is_verbose
            )

            subfolder_name.mkdir(exist_ok=True)
            torch.save(train_loss_log, subfolder_name/'train_loss.pth')
            torch.save(val_loss_log, subfolder_name/'val_loss.pth')
            torch.save(net, subfolder_name/'net.pth')

            if self.is_verbose:
                print(f'{net_name} training finished')

    def plot_loss(self) -> None:
        for group in ['train', 'val']:
            _, ax = plt.subplots(figsize=(8, 6))
            y_min, y_max = np.inf, -np.inf
            for net_name in ['ReLU', 'ELU', 'GELU', 'SLU (shared)', 'SLU (individual)']:
                subfolder_name = self.artifacts_dir/(net_name.replace(' ', '_'))
                loss = torch.load(subfolder_name/f'{group}_loss.pth')
                loss = smooth_data(loss, self.window_size)

                y_min = min(y_min, min(loss))
                y_max = max(y_max, np.percentile(loss, 95))

                x = np.linspace(0, self.n_epochs, len(loss))
                plt.plot(x, loss, c=COLORS[net_name], label=f'{net_name}', alpha=0.8)
                plt.axhline(min(loss), linewidth=2, linestyle='dashed', c=COLORS[net_name], alpha=0.8)

            plt.yscale('log')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
            ax.yaxis.set_minor_formatter(FormatStrFormatter('%.3g'))

            plt.xlim([0, self.n_epochs])
            plt.ylim([y_min - 5e-4, y_max])

            plt.xlabel('Epoch')
            plt.ylabel(f'Log Loss (lr = {to_scientific_notation(self.lr)})')

            plt.grid(which='both')
            plt.axhline(y_min - 10, linewidth=2, linestyle='dashed', c='k', label='Minimal loss levels')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.artifacts_dir/f'{group}_comparison.png')
            plt.clf()

    def plot(self) -> None:
        self.plot_loss()

class SummarizeCIFAR10(BaseExperiment):
    def __init__(self, artifacts_dir: Directory, n_epochs: int = 20, window_size: int = 128):
        super().__init__(artifacts_dir, Preprocessor)
        self.window_size = window_size
        self.n_epochs = n_epochs

    def plot_best_scores(self) -> None:
        _, ax = plt.subplots(figsize=(8, 6))
        markers = ['o', '^', 'X', 'd', 'p', 'P', 'v', '*']
        handles = []

        for net_name in ['ReLU', 'ELU', 'GELU', 'SLU (shared)', 'SLU (individual)']:
            handles.append(mpatches.Patch(label=net_name, color=COLORS[net_name]))

        marker_i = 0
        for item in Path(self.artifacts_dir).iterdir():
            if not item.is_dir(): continue
            architecture_name = item.name
            handles.append(mlines.Line2D([], [], color='black', marker=markers[marker_i], linestyle='None',
                           markersize=10, label=architecture_name))
            marker_i += 1

        handles.append(mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                       markersize=15, label='Average'))
        handles.append(mlines.Line2D([], [], color='black', marker='', label='', alpha=0))

        with open(self.artifacts_dir/f'mean_best_scores.txt', 'wt') as f:
            for net_name in ['ReLU', 'ELU', 'GELU', 'SLU (shared)', 'SLU (individual)']:
                min_iter_list = []
                min_loss_list = []
                marker_i = 0
                for item in Path(self.artifacts_dir).iterdir():
                    if not item.is_dir(): continue
                    architecture_name = item.name

                    subfolder_name = self.artifacts_dir/architecture_name/(net_name.replace(' ', '_'))
                    loss = torch.load(subfolder_name/'val_loss.pth')
                    loss = smooth_data(loss, self.window_size)

                    x = np.linspace(0, self.n_epochs, len(loss))
                    min_iter = x[np.argmin(loss)]
                    min_loss = np.min(loss)

                    f.write(f'{net_name}; {architecture_name}; {min_iter}; {min_loss}\n')
                    min_iter_list.append(min_iter)
                    min_loss_list.append(min_loss)
                    plt.scatter(min_iter, min_loss, c=COLORS[net_name], s=80, marker=markers[marker_i])
                    marker_i += 1

                mean_iter, mean_loss = np.mean(min_iter_list), np.mean(min_loss_list)
                plt.scatter(mean_iter, mean_loss, c=COLORS[net_name], s=100,
                            marker='s', edgecolors='k', linewidth=3)

                f.write(f'\t{net_name}; {mean_iter}; {mean_loss}\n')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.xlabel('Epoch')
        plt.ylabel(f'Log Loss')

        plt.grid(which='both')

        plt.legend(handles=handles, ncol=2)
        plt.tight_layout()
        plt.savefig(self.artifacts_dir/f'best_scores.png')
        plt.clf()

    def train(self) -> None:
        pass

    def plot(self) -> None:
        self.plot_best_scores()
