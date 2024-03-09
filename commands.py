from pathlib import Path

import fire

from src.visualizations import (
    PlotActivations,
    PlotSLU
)
from src.experiments import (
    Classify2D
)

log_dir = Path('logs')


def plot_activations() -> None:
    PlotActivations(fig_path=log_dir/'activations.png').evaluate()


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


if __name__ == "__main__":
    fire.Fire()

