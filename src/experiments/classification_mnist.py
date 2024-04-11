import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import torch.utils.data as data
from torchvision import datasets, transforms

from ..colors import COLORS
from ..layers import SLU
from ..types import Callable, Directory, Tuple
from ..utils import smooth_data, to_scientific_notation, train
from .base import BaseExperiment, BasePreprocessor


np.random.seed(0)
torch.manual_seed(0)


class Preprocessor(BasePreprocessor):
    '''NOTE:
    No preprocessor is needed
    as pytorch has a prebuilt MNIST integration
    with torchvision transforms support
    '''
    def __init__(self):
        super().__init__()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def transform(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return X, y


class ClassifyMNIST(BaseExperiment):
    def __init__(self, artifacts_dir: Directory,
                 n_layers: int, n_neurons: int, force_retrain: bool,
                 noise: float = 0,
                 n_epochs: int = 20, lr: float = 1e-3,
                 is_verbose: bool = True, window_size: int = 128):
        super().__init__(artifacts_dir, Preprocessor)
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.force_retrain = force_retrain
        self.noise = noise
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_verbose = is_verbose
        self.window_size = window_size

    def init_network(self, activation_fn: Callable[[int], nn.Module]) -> nn.Sequential:
        layers = [
            nn.Linear(784, self.n_neurons),
            activation_fn(self.n_neurons)
        ]
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(activation_fn(self.n_neurons))
        layers.append(nn.Linear(self.n_neurons, 10))
        return nn.Sequential(*layers)

    def train(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten),
            transforms.Lambda(lambda x: x + torch.randn(x.shape) * self.noise)
        ])

        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('data', train=False, transform=transform)
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

