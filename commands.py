from pathlib import Path

import fire

from src.visualizations import (
    PlotActivations,
    PlotSLU
)
from src.experiments import (
    Classify2D,
    Regress1D
)

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)


def plot_activations() -> None:
    PlotActivations(fig_path=log_dir/'activations.png').evaluate()
    PlotActivations(fig_path=log_dir/'sigmoid.png', function='Sigmoid').evaluate()
    PlotActivations(fig_path=log_dir/'relu.png', function='ReLU').evaluate()
    PlotActivations(fig_path=log_dir/'elu.png', function='ELU').evaluate()
    PlotActivations(fig_path=log_dir/'gelu.png', function='GELU').evaluate()


def plot_SLU() -> None:
    PlotSLU(x_min=-20, x_max=20, fig_path=log_dir/'slu_20.png').evaluate()
    PlotSLU(x_min=-3, x_max=3, fig_path=log_dir/'slu_3.png').evaluate()


def classify_spirals(train: bool = True) -> None:
    experiment = Classify2D(artifacts_dir=log_dir/'spirals', dataset='spirals')
    if train: experiment.train()
    experiment.plot()


def classify_moons(train: bool = True) -> None:
    experiment = Classify2D(artifacts_dir=log_dir/'moons', dataset='moons')
    if train: experiment.train()
    experiment.plot()


def regress_1d_square(train: bool = True) -> None:
    experiment = Regress1D(artifacts_dir=log_dir/'square', dataset='square')
    if train: experiment.train()
    experiment.plot()


def regress_1d_root(train: bool = True) -> None:
    experiment = Regress1D(artifacts_dir=log_dir/'root', dataset='root')
    if train: experiment.train()
    experiment.plot()


def regress_1d_reciprocal(train: bool = True) -> None:
    experiment = Regress1D(artifacts_dir=log_dir/'reciprocal', dataset='reciprocal')
    if train: experiment.train()
    experiment.plot()


if __name__ == '__main__':
    fire.Fire()

